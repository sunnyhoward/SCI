import torch
from torch.utils.data import Dataset
import h5py
import forward as fwd
import time
import numpy as np
from skimage.transform import rotate
import os, sys
main_dir = os.path.dirname(os.path.dirname(os.path.abspath('')))
sys.path.insert(0, main_dir)
from models.custom.modules import CenterOfMassLoss
import scipy


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='cuda'):
    '''
    The function to train a model.

    @TODO: Add the ability to redirect the print statements to the model_results.txt file.
    '''

    try:  model_name = model.name
    except: model_name = type(model).__name__
    lr = optimizer.param_groups[0]['lr']

    print('train called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (model_name, type(optimizer).__name__,
           lr, epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    total_batches = int(len(train_dl.indexes)/train_dl.batch_size)

    start_time_sec = time.time()

    for epoch in range(1, epochs+1):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
        start_time_sec = time.time()
        model.train()
        train_loss         = 0.0
        num_train_examples = 0

        for n,(x,y) in enumerate(train_dl):

            optimizer.zero_grad()

            x    = x.to(device,dtype=torch.float32)
            y    = y.to(device,dtype=torch.float32)
            yhat = model(x)
            loss = loss_fn(yhat, y)

            loss.backward()
            optimizer.step()

            train_loss         += loss.data.item() * x.size(0)
            # num_train_correct  += (torch.max(yhat, 1)[1] == y).sum().item()
            num_train_examples += x.shape[0]

            print(f'Epoch {epoch}/{epochs}: ({n}/{total_batches}), train loss: {(train_loss/num_train_examples):5.5g}',end='\r')

        train_loss  = train_loss / len(train_dl.indexes)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
        with torch.no_grad():
            val_loss       = 0.0
            num_val_examples = 0

            for n,(x,y) in enumerate(val_dl):

                x    = x.to(device,dtype=torch.float32)
                y    = y.to(device,dtype=torch.float32)
                yhat = model(x)
                loss = loss_fn(yhat, y)

                val_loss         += loss.data.item() * x.size(0)
                num_val_examples += y.shape[0]

            val_loss = val_loss / len(val_dl.indexes)

        end_time_sec       = time.time()
        total_time_sec     = end_time_sec - start_time_sec

        print(f'Epoch {epoch}/{epochs}: ({total_batches}/{total_batches}), train loss: {train_loss:5.5g}, val_loss = {val_loss:5.5g},  epoch time: {total_time_sec:5.5g}')


        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        
    with open('model_results.txt', 'a') as file:
        file.write(f'model={model_name}, lr = {lr}, epochs = {epochs}\n')
        file.write(f'epoch time: {total_time_sec}\n')
        file.write(f'history: {history}\n')
        file.write('\n\n\n')

    return history






class CustomDataLoader:
    def __init__(self, dataset, indexes, batch_size=1, shuffle=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = indexes #this allows us to have a train and vali set.
        self.dataset.set_batch(batch_size)

    def __iter__(self):

        indices = self.indexes[torch.randperm(len(self.indexes))] if self.shuffle else self.indexes

        for batch_start in range(0, len(indices), self.batch_size):
            batch_indices = indices[batch_start:batch_start + self.batch_size]

            x,y = self.dataset[batch_indices]
            yield x,y

    def __len__(self):
        return len(self.dataset) // self.batch_size



class SyntheticDataset(Dataset):
    '''
    Generate data for training by taking an undispersed cube, dispersing it and applying spectral modulation.

            Parameters:
                    undispersed_cube (array): This should really be the imaged mask cube
                    shift_info (dict): contains either dispersion array or kernel.
                    spectra (array): the spectral modulation information from fts

            Returns:
                    (array): undispersed, unintegrated cube
    '''
    def __init__(self, undispersed_cube, shift_info, spectra, crop=False, random_shifts=False):
        super(SyntheticDataset, self).__init__()

        self.undispersed_cube = undispersed_cube
        self.shift_info = shift_info
        self.spectra = spectra
        self.crop = crop    
        self.sensing_function = fwd.fourier.method.calc_psi_z 
        self.random_shifts = random_shifts
        self.set_batch()


    def set_batch(self,batch=2):
        self.batch = batch
        self.data = torch.tile(self.undispersed_cube,(batch,1,1,1))

    def __len__(self):
        return len(self.spectra[0])

    def __getitem__(self, index):

        x = self.spatiotemporal_mult(self.data, index) #apply a modulation to the cube

        if self.random_shifts:

            roll_i, roll_j = np.random.rand() * 3 - 1.5, np.random.rand() * 3 - 1.5

            x_upup = torch.roll(x, shifts=(int(np.ceil(roll_i)),int(np.ceil(roll_j))), dims=(2, 3))
            x_downdown = torch.roll(x, shifts=(int(np.floor(roll_i)),int(np.floor(roll_j))), dims=(2, 3))
            x_updown = torch.roll(x, shifts=(int(np.ceil(roll_i)),int(np.floor(roll_j))), dims=(2, 3))
            x_downup = torch.roll(x, shifts=(int(np.floor(roll_i)),int(np.ceil(roll_j))), dims=(2, 3))

            x = x_upup * (roll_i - np.floor(roll_i)) * (roll_j - np.floor(roll_j)) + x_downdown * (np.ceil(roll_i) - roll_i) * (np.ceil(roll_j) - roll_j) + x_updown * (np.ceil(roll_i) - roll_i) * (roll_j - np.floor(roll_j)) + x_downup * (roll_i - np.floor(roll_i)) * (np.ceil(roll_j) - roll_j)


        y = self.sensing_function(torch.ones_like(x),x,shift_info=self.shift_info) #measure it

        y[y<0] = 0

        if self.crop != False:
            nx,ny = x.shape[2:]
            cropx = self.crop[0] // 2
            cropy = self.crop[1] // 2

            x = x[...,nx//2 - cropx : nx//2+cropx,ny//2 - cropy : ny//2+cropy]
        
        return y, x
    

    def spatiotemporal_mult(self, mask_cube, index, type = 'spectra'):
        '''
        Take the relay-imaged mask_cube (bs,nc,nx,ny) and multiply it by some spatiotemporal map. 

        type: 'spectra' or 'seperable' or 'spatiotemporal'
        '''
        nx,ny = mask_cube.shape[2:]


        if type == 'spectra':
            x = mask_cube * self.spectra[:,index].permute(1,0).unsqueeze(-1).unsqueeze(-1)

        elif type == 'seperable': #not implemented yet
            x = mask_cube * self.spectra[:,index].permute(1,0).unsqueeze(-1).unsqueeze(-1) 
            #... x[...,nx//2 - 320 : nx//2+320,ny//2 - 320 : ny//2+320] = x[...,nx//2 - 320 : nx//2+320,ny//2 - 320 : ny//2+320] * self.spatial_intensity[

        elif type == 'spatiotemporal':
            x = mask_cube[...,nx//2 - 320 : nx//2+320,ny//2 - 320 : ny//2+320] * self.spatiotemporal[index] #not really implemented yet either.

        return x
        


class FTSDataset(Dataset):
    '''
    Here we generate data from the real FTS measurements. 

            Parameters:
                    undispersed_cube (array): This should really be the imaged mask cube
                    spectra (array): the spectral modulation information from fts

            Returns:
                    (array): undispersed, unintegrated cube
    '''
    def __init__(self, undispersed_cube, spectra, positions, dir, crop=False,random_shifts=False):
        super(FTSDataset, self).__init__()

        self.undispersed_cube = undispersed_cube
        self.crop = crop
        self.random_shifts = random_shifts
        self.set_batch()
        self.dir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'+dir + '/'
        self.positions = positions# np.load(self.dir+'/positions.npy')[indices]
        self.spectra = spectra #np.load(self.dir+'/spectra.npy')[:,indices]
        # self.angle = angle

        
        
        try: self.integration_time =   int(dir.split('0us')[0].split('_')[-1] + '0'); 
        except: self.integration_time =   int(dir.split('0_us')[0].split('_')[-1] + '0'); 


    def set_batch(self,batch=2):
        self.batch = batch
        self.data = torch.tile(self.undispersed_cube,(batch,1,1,1))

    def __len__(self):
        return len(self.spectra[0])

    def __getitem__(self, index):

        x = self.data * self.spectra[:,index].permute(1,0).unsqueeze(-1).unsqueeze(-1)
        # y = torch.stack([torch.from_numpy( rotate(np.load(self.dir + 'piezopos_' + str(i) + '.npy').astype(np.float32), angle=self.angle )) for i in self.positions[index]],dim = 0) # - 0.0002 # removing background noise?
        y = torch.stack([torch.from_numpy( np.load(self.dir + 'piezopos_' + str(i) + '.npy').astype(np.float32)) for i in self.positions[index]],dim = 0) # - 0.0002 # removing background noise?

   
        y = y/self.integration_time

        if self.random_shifts:
            roll_i, roll_j = np.random.randint(-7,7), np.random.randint(-7,7)

            x = torch.roll(x, shifts=(roll_i,roll_j), dims=(2, 3))
            y = torch.roll(y, shifts=(roll_i,roll_j), dims=(1, 2))

        if self.crop != False:
            nx,ny = x.shape[2:]
            cropx = self.crop[0] // 2
            cropy = self.crop[1] // 2

            x = x[...,nx//2 - cropx : nx//2+cropx,ny//2 - cropy : ny//2+cropy]

        
        return y, x
    




class HDF5Dataset(Dataset):
    def __init__(self, hdf5_file_name, keys=['data', 'labels']):
        super(HDF5Dataset, self).__init__()

        hdf5_file = h5py.File(hdf5_file_name, 'r')
        self.data = hdf5_file[keys[0]]
        self.labels = hdf5_file[keys[1]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index])  # Convert data to a PyTorch tensor
        y = torch.tensor(self.labels[index])  # Convert labels to a PyTorch tensor
        return x, y
    






###########################################################################################


def create_bs_data(desired_channels, fts_dir, desired_range=[700,900], cube_dir = None, interp_type='average', device='cuda'):
    '''
    This is ugly so put it in a function. 

    Calling this function will yield the undispersed cube, the dispersed mask and the FTS spectra.

    desired_channels is the number of channels you want to interpolate to.
    kernel is the kernel to disperse with.
    fts_dir is the directory of the fts data (with grating)
    cube_dir is the directory of the undispersed cube (maybe without grating)
    interp_type is the type of interpolation. Can be 'nearest'(fast) or 'average'(slow)
    crop_cube is whether to crop the cube to 640x640 or not (useful when we have a 9 copy fts data.)
    '''
    print('collecting the undispersed cube and spectra.')


    datadir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'+fts_dir + '/'
    

    

    ###load the cube and FTS spectra###
    if cube_dir!=None:
        cube_dir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'+cube_dir + '/'


        undisp_cube = load_cube(cube_dir)
        undisp_cube = downsample_signal(undisp_cube, desired_channels, initial_bins, interp_axis = 2, desired_range = desired_range, interp_type=interp_type) #interpolate to 750-850nm
        undisp_cube = torch.tensor(undisp_cube).float().permute(2,0,1).unsqueeze(0).to(device)
        try: initial_bins = np.load(cube_dir+'wavls_padded_fixed_thresholded.npy')
        except: 
            try: 
                initial_bins = np.load(cube_dir+'wavls_padded_fixed.npy')
            except: initial_bins = np.load(cube_dir+'wavls_padded.npy')
    else:
        undisp_cube=None
    

    ###interpolate to desired channels###
    
    spectras = np.load(datadir+'spectra.npy') 
    initial_bins = np.linspace(634.69, 1124.5, spectras.shape[0])*1e-9
    spectras = downsample_signal(spectras, desired_channels, initial_bins, desired_range = desired_range, interp_axis = 0, interp_type=interp_type) #interpolate to 750-850nm

    
    ###normalize and send to cuda###
    
    spectras = torch.tensor(spectras).float().to(device)
    spectras = (spectras - spectras.min()) / (spectras.max() - spectras.min()) #normalize the spectra individually between 0 and 1
    
    return undisp_cube, spectras#dispersed_mask


# def threshold(data):
#     # return data
#     datamax = np.max(data)
#     data[data<0.001*datamax] = 0
#     return data


def normalize(data, integration_time):
    # return data
    return data / integration_time/ 5
    # return (data - np.min(data)) / (np.max(data) - np.min(data))



def downsample_signal(data, desired_channels, initial_bins, desired_range = [700,900], interp_axis = -1, interp_type='average'):
    '''
    TO BE RENAMED to DOWNSAMPLE_SIGNAL
    simplest possible example.
    desired_channels is desired channels.
    interp_axis is the axis to interpolate on.
    interp_type is the type of interpolation. Can be 'nearest'(fast) or 'average'(slow)
    '''

    if interp_axis==-1:    interp_axis = len(data.shape)-1
    
    desired_bins = np.linspace(desired_range[0],desired_range[1],desired_channels)*1e-9

    # f = scipy.interpolate.interp1d(initial_bins, data, axis=interp_axis)

    # newdata = f(desired_bins)
    
    if interp_type == 'nearest':
        idx = np.zeros_like(desired_bins,dtype=int)
        for i in np.arange(len(desired_bins)):
            idx[i] = (np.abs(initial_bins - desired_bins[i])).argmin() #just find the closest one. (can replace with mean or something)
        newdata = np.take(data,idx,axis=interp_axis)

    elif interp_type == 'average':
        newshape = list(data.shape)
        newshape[interp_axis] = desired_channels
        newdata = np.zeros(newshape)

        dlambda = (desired_bins[1] - desired_bins[0])/2

        for i in np.arange(desired_channels):
            idx = np.where((initial_bins > desired_bins[i] - dlambda) * (initial_bins<desired_bins[i]+dlambda)) [0]
            indices = tuple([i if j==interp_axis else slice(None) for j in range(len(newshape))])
            newdata[indices] = np.mean(np.take(data,idx,axis=interp_axis),axis=interp_axis)

    return newdata




def load_cube(anadir, postfix = '', onefile=True):

    anadir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'+anadir + '/'

    if '0us' in anadir: integration_time = int(anadir.split('0us')[0].split('_')[-1] + '0')
    elif '0_us' in anadir: integration_time = int(anadir.split('0_us')[0].split('_')[-1] + '0')

    if onefile:
        undisp_cube = np.load(anadir+'signalfft_'+postfix+'.npy')

        wavls = np.load(anadir+'wavls_'+postfix+'.npy')
        freqs = np.load(anadir+'freqs_'+postfix+'.npy')
    else: #the postfix is for the center.

        signalfft_center = np.load(anadir+'signalfft_'+postfix+'.npy')
        left_string = postfix.split('center')[0] + 'left' + postfix.split('center')[1]
        right_string = postfix.split('center')[0] + 'right' + postfix.split('center')[1]

        signalfft_left = np.load(anadir+'signalfft_'+left_string+'.npy')
        signalfft_right = np.load(anadir+'signalfft_'+right_string+'.npy')
        undisp_cube = np.concatenate((signalfft_left,signalfft_center,signalfft_right),axis = 1) 

        nostring = postfix.split('center_')[0] + postfix.split('center_')[1]
        wavls = np.load(anadir+'wavls_'+nostring+'.npy')
        freqs = np.load(anadir+'freqs_'+nostring+'.npy')

    undisp_cube = normalize(undisp_cube, integration_time)
    
    return undisp_cube, wavls, freqs


def load_spectra(fts_dir,desired_channels, desired_range, device):

    datadir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'+fts_dir + '/' 
    
    spectras = np.load(datadir+'spectra.npy') 
    initial_bins = np.linspace(634.69, 1124.5, spectras.shape[0])

    spectras, final_bins = fix_spectra(spectras, initial_bins, desired_range)

    spectras = (spectras - np.min(spectras,axis=0)) / (np.max(spectras,axis=0) - np.min(spectras,axis=0)) #normalize the spectra individually between 0 and 1

    spectras = downsample_signal(spectras, desired_channels, final_bins * 1e-9, desired_range = desired_range, interp_axis = 0, interp_type='average') #interpolate to 750-850nm

    
    ###normalize and send to cuda###
    
    spectras = torch.tensor(spectras).float().to(device)
    
    return spectras


def fix_spectra(spectras, initial_bins, desired_range, verbose=False):
    # the spectras have the intensity of the signal overlayed. We wish to remove that.
    spectras = spectras[((initial_bins>desired_range[0]) & (initial_bins < desired_range[1]))]
    final_bins = initial_bins[((initial_bins>desired_range[0]) & (initial_bins < desired_range[1]))]


    mean_spectra = np.mean(spectras,axis=1)
    normed_spectra = spectras / mean_spectra[:,None]

    fft_spectra = np.abs(np.fft.fftshift(np.fft.fft(normed_spectra, axis=0),axes=0))
    length = fft_spectra.shape[1]
    bla = np.fft.fftshift(np.fft.fftfreq(len(normed_spectra), d=1))

    # plt.figure(dpi=250)
    # plt.imshow(fft_spectra, extent = [0,length,bla[0],bla[-1]],aspect='auto'); plt.xlabel('shotno'); plt.ylabel('frequency')

    threshold = 0.2
    from scipy import ndimage

    uber0 = fft_spectra[bla>0]
    uber0[uber0<uber0.max()*threshold] = 0
    ele = ndimage.center_of_mass(uber0[:,0])[0]
    init = bla[bla>0][np.floor(ele).astype(int)] * (1-ele%1) + bla[bla>0][np.ceil(ele).astype(int)] * (ele%1)

    unter0 = fft_spectra[bla<0]
    unter0[unter0<unter0.max()*threshold] = 0
    ele = ndimage.center_of_mass(unter0[:,0])[0]
    final = bla[bla<0][np.floor(ele).astype(int)] * (1-ele%1) + bla[bla<0][np.ceil(ele).astype(int)] * (ele%1)

    # plt.plot([0,length],[init,final], 'r', linewidth=0.1)

    frequency = np.arange(length) / length * (final - init) + init

    def func(x, a, b, c, d):
        return a + b * np.sin(2*np.pi* c * x + d)
    xpos = np.arange(len(normed_spectra))

    pred_spectras = np.zeros_like(normed_spectra)

    from scipy.optimize import curve_fit


    for i in range(len(normed_spectra[0])):
        p0 = [1., 0.5, frequency[i], 0.]
        y = normed_spectra[:,i]
        try:
            p1, var_matrix = curve_fit(func, xpos, y, p0)
            # p_ult = np.array([1, p1[1], p1[2], p1[3]])
            pred_spectras[:,i] = func(xpos, *p1)
        except:
            pred_spectras[:,i] = pred_spectras[:,i-1]

    pred_spectras = (pred_spectras - np.min(pred_spectras,axis=0)) / (np.max(pred_spectras,axis=0) - np.min(pred_spectras,axis=0)) #normalize the spectra individually between 0 and 1


    return pred_spectras, final_bins
        


def center_cubes(nograting_cube, grating_cube,  device='cpu'):
    '''
    given a cube without grating and one with grating, match the locations of the fundamental.
    '''

    #first center the position of the fundamental in the grating cube
    sx,sy = grating_cube.shape[2:]

    center_x = torch.sum(grating_cube[0,:,sx//2-200:sx//2+200,sy//2-400:sy//2+400],dim=(0,2)).argmax() + sx//2-200
    center_y  = torch.sum(grating_cube[0,:,sx//2-200:sx//2+200,sy//2-400:sy//2+400],dim=(0,1)).argmax() + sy//2-400

    rollx = (center_x-sx//2,center_y-sy//2)
    print('to center grating we move by = ' + str(rollx))

    pointspot = sx//2  + rollx[0]   , sy//2+ rollx[1]

    grating_cube = torch.roll(grating_cube, shifts=(-rollx[0],-rollx[1]), dims=(2, 3)) # shift them both to the center of grating_cube
    nograting_cube = torch.roll(nograting_cube, shifts=(-rollx[0],-rollx[1]), dims=(2, 3))

    # if shift_ng_to_g:

        # nograting_cube = shift_nograting_to_grating(nograting_cube, grating_cube, y_range=[0,500], device=device)


    return nograting_cube, grating_cube



def shift_funda_of_grating(grating_cube, wl_range, y_range):
    #for some reason there seems to be a problem with the fundamental 
    CoM_y_l, CoM_x_l = CenterOfMassLoss.calculate_center_of_mass(grating_cube[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[0,700]])).mean(0).cpu()

    CoM_y_c, CoM_x_c = CenterOfMassLoss.calculate_center_of_mass(grating_cube[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[1200,1300]])).mean(0).cpu()
    CoM_y_c, CoM_x_c = CoM_y_c + y_range[0], CoM_x_c + 1200

    CoM_y_r, CoM_x_r = CenterOfMassLoss.calculate_center_of_mass(grating_cube[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[1550,2250]])).mean(0).cpu()
    CoM_y_r, CoM_x_r = CoM_y_r + y_range[0], CoM_x_r + 1550
    print(CoM_y_r, CoM_y_l, CoM_y_c)


    shiftit =   (CoM_y_r + CoM_y_l) // 2  - CoM_y_c

    print('to shift funda of grating we move by = ' + str(shiftit.numpy()))


    grating_funda = grating_cube[:,:,y_range[0]:y_range[1], int(CoM_x_c)-200:int(CoM_x_c)+200]
    
    grating_funda = torch.tensor(scipy.ndimage.shift(grating_funda.cpu(), shift = (0,0,shiftit,0), order = 1)).to(device)


    grating_cube[:,:,y_range[0]:y_range[1], int(CoM_x_c)-200:int(CoM_x_c)+200] = grating_funda

    CoM_y_c, CoM_x_c = CenterOfMassLoss.calculate_center_of_mass(grating_cube[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[1200,1300]])).mean(0).cpu()
    CoM_y_c, CoM_x_c = CoM_y_c + y_range[0], CoM_x_c + 1200
    print(CoM_y_c, CoM_x_c)

    return grating_cube

def sub_nogratingfunda(nograting_cube,grating_cube):
    sx,sy = nograting_cube.shape[-2:]
    spatial_funda = grating_cube[:,:,sx//2-200:sx//2+200, sy//2-200:sy//2+200]

    ngspectra = torch.mean(nograting_cube[:,:,sx//2-200:sx//2+200, sy//2-200:sy//2+200],dim=(0,2,3)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    gspectra = torch.mean(grating_cube[:,:,sx//2-200:sx//2+200, sy//2-200:sy//2+200],dim=(0,2,3)).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    nograting_cube[...,sx//2-200:sx//2+200,sy//2-200:sy//2+200] = spatial_funda*ngspectra/gspectra
    return nograting_cube


def shift_nograting_to_grating(nograting_cube, grating_cube, y_range, device='cpu'):


    region = np.array([[y_range[0],y_range[1]],[700,1600]])
    CoM_nog = CenterOfMassLoss.calculate_center_of_mass(nograting_cube[0],region).mean(0)
    CoM_g = CenterOfMassLoss.calculate_center_of_mass(grating_cube[0],region).mean(0)

    # shiftdown = torch.floor(CoM_g - CoM_nog).cpu().int()
    # shiftup = torch.ceil(CoM_g - CoM_nog).cpu().int()

    shifts = (CoM_g - CoM_nog).cpu()

    print('to shift nograting to grating we move by = ' + str(shifts.numpy()))

    result = torch.tensor(scipy.ndimage.shift(nograting_cube.cpu(), shift = (0,0,shifts[0],shifts[1]), order = 1)).to(device)

    return result# (torch.roll(nograting_cube,tuple(shiftdown.numpy()),dims=(2,3)) + torch.roll(nograting_cube,tuple(shiftup.numpy()),dims=(2,3))) / 2



def remove_rotation(kernel, y_range = [0,100], wl_range=[0,41], verbose=False):
    
    mid = y_range[0] + (y_range[1]-y_range[0])//2

    if verbose:
        fig,ax = plt.subplots(1,3,dpi=200)
        ax[0].imshow(torch.sum((kernel)[0,wl_range[0]:wl_range[1]],dim=0).cpu().detach().numpy()[y_range[0]:y_range[1],250:450],vmax=0.00001)
        ax[1].imshow(torch.sum((kernel)[0,wl_range[0]:wl_range[1]],dim=0).cpu().detach().numpy()[y_range[0]:y_range[1], 1200:1300],vmax=0.00001)
        ax[2].imshow(torch.sum((kernel)[0,wl_range[0]:wl_range[1]],dim=0).cpu().detach().numpy()[y_range[0]:y_range[1], 2000:2200],vmax=0.00001)
        ax[0].plot(np.arange(100),np.ones(100)*mid,'r',markersize=0.05)
        ax[1].plot(np.arange(100),np.ones(100)*mid,'r',markersize=0.05)
        ax[2].plot(np.arange(100),np.ones(100)*mid,'r',markersize=0.05)



    CoM_y_l, CoM_x_l = CenterOfMassLoss.calculate_center_of_mass(kernel[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[0,700]])).mean(0).cpu()
    print(CoM_x_l,CoM_y_l)
    CoM_y_c, CoM_x_c = CenterOfMassLoss.calculate_center_of_mass(kernel[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[1200,1300]])).mean(0).cpu()
    CoM_y_c, CoM_x_c = CoM_y_c + y_range[0], CoM_x_c + 1200
    print(CoM_x_c,CoM_y_c)
    CoM_y_r, CoM_x_r = CenterOfMassLoss.calculate_center_of_mass(kernel[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[1550,2250]])).mean(0).cpu()
    CoM_y_r, CoM_x_r = CoM_y_r + y_range[0], CoM_x_r + 1550
    print(CoM_x_r,CoM_y_r)

    angle = np.arctan((CoM_y_r-CoM_y_l)/(CoM_x_r-CoM_x_l))

    from skimage.transform import rotate

    rot_kernel = torch.tensor(np.stack([rotate(kernel[0,i].cpu().numpy(),angle=angle*180/np.pi) for i in range(kernel.shape[1])])).unsqueeze(0)

    if verbose:
        fig,ax = plt.subplots(1,3,dpi=200)
        ax[0].imshow(torch.sum((rot_kernel)[0],dim=0).cpu().detach().numpy()[y_range[0]:y_range[1],150:450],vmax=0.00001)
        ax[1].imshow(torch.sum((rot_kernel)[0],dim=0).cpu().detach().numpy()[y_range[0]:y_range[1], 1200:1300],vmax=0.00001)
        ax[2].imshow(torch.sum((rot_kernel)[0],dim=0).cpu().detach().numpy()[y_range[0]:y_range[1], 2000:2300],vmax=0.00001)
        ax[0].plot(np.arange(100),np.ones(100)*mid,'r',markersize=0.05)
        ax[1].plot(np.arange(100),np.ones(100)*mid,'r',markersize=0.05)
        ax[2].plot(np.arange(100),np.ones(100)*mid,'r',markersize=0.05)

    CoM_y_l, CoM_x_l = CenterOfMassLoss.calculate_center_of_mass(rot_kernel[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[0,700]])).mean(0).cpu()
    print(CoM_x_l,CoM_y_l)
    CoM_y_c, CoM_x_c = CenterOfMassLoss.calculate_center_of_mass(rot_kernel[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[1200,1300]])).mean(0).cpu()
    CoM_y_c, CoM_x_c = CoM_y_c + y_range[0], CoM_x_c + 1200
    print(CoM_x_c,CoM_y_c)
    CoM_y_r, CoM_x_r = CenterOfMassLoss.calculate_center_of_mass(rot_kernel[0,wl_range[0]:wl_range[1]], region = np.array([[y_range[0],y_range[1]],[1550,2250]])).mean(0).cpu()
    CoM_y_r, CoM_x_r = CoM_y_r + y_range[0], CoM_x_r + 1550
    print(CoM_x_r,CoM_y_r)


    return rot_kernel

# kernel = remove_rotation(kernel, verbose=True).to(device)
