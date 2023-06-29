import torch
from torch.utils.data import Dataset
import h5py
import forward as fwd
import time
import numpy as np


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
    def __init__(self, undispersed_cube, shift_info, spectra, method='fourier', crop=True):
        super(SyntheticDataset, self).__init__()

        self.undispersed_cube = undispersed_cube
        self.shift_info = shift_info
        self.spectra = spectra
        self.crop = crop    
        self.sensing_function = fwd.fourier.method.calc_psi_z if method == 'fourier' else fwd.rolling.method.calc_psi_z
        self.set_batch()


    def set_batch(self,batch=2):
        self.batch = batch
        self.data = torch.tile(self.undispersed_cube,(batch,1,1,1))

    def __len__(self):
        return len(self.spectra[0])

    def __getitem__(self, index):

        x = self.spatiotemporal_mult(self.data, index) #apply a modulation to the cube

        y = self.sensing_function(torch.ones_like(x),x,shift_info=self.shift_info) #measure it
        
        if self.crop:
            nx,ny = x.shape[2:]
            x = x[...,nx//2 - 320 : nx//2+320,ny//2 - 320 : ny//2+320]
        
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
    def __init__(self, undispersed_cube, spectra, dir, crop=True):
        super(FTSDataset, self).__init__()

        self.undispersed_cube = undispersed_cube
        self.spectra = spectra
        self.crop = crop    
        self.set_batch()
        self.dir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'+dir + '/'
        self.positions = np.load(self.dir+'/positions.npy')


    def set_batch(self,batch=2):
        self.batch = batch
        self.data = torch.tile(self.undispersed_cube,(batch,1,1,1))

    def __len__(self):
        return len(self.spectra[0])

    def __getitem__(self, index):

        x = self.data * self.spectra[:,index].permute(1,0).unsqueeze(-1).unsqueeze(-1)
        y = torch.stack([torch.from_numpy(np.load(self.dir + 'piezopos_' + str(i) + '.npy').astype(np.float32)/ 4096) for i in self.positions[index]],dim = 0) 

        if self.crop:
            nx,ny = x.shape[2:]
            x = x[...,nx//2 - 320 : nx//2+320,ny//2 - 320 : ny//2+320]
        
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


def create_bs_data(desired_channels, kernel, fts_dir, cube_dir = None, interp_type='average', device='cuda'):
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

    ###load the cube###
    anasubdir = fts_dir.split('data')
    anasubdir = anasubdir[0] + 'analysis' + anasubdir[1]

    anadir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'+anasubdir + '/'
    datadir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'+fts_dir + '/'
    
    if cube_dir is None:
        cube_dir = anadir
        cube_dir_stat = 'copy'
    else:
        cube_dir  = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/'+cube_dir + '/'
        cube_dir_stat = 'real'

    ###load the cube and FTS spectra###
    undisp_cube = load_cube(cube_dir)
    spectras = np.load(datadir+'spectra.npy') 

    ###interpolate to desired channels###
    undisp_cube = downsample_signal(undisp_cube, desired_channels, interp_axis = 2, interp_type=interp_type) #interpolate to 750-850nm
    spectras = downsample_signal(spectras, desired_channels, initial_range=[345.114169, 1036.5037173765845], interp_axis = 0, interp_type=interp_type) #interpolate to 750-850nm

    #if cube_dir is copied, then we should just take the central region of it as the true cube (the undispersed)
    if cube_dir_stat == 'copy':
        zeros = np.zeros_like(undisp_cube)
        sx,sy = undisp_cube.shape[:2]
        zeros[sx//2 - 320 : sx//2+320,sy//2 - 320 : sy//2+320] = undisp_cube[sx//2 - 320 : sx//2+320,sy//2 - 320 : sy//2+320]
        undisp_cube = zeros


    try:
        dispersed_mask = torch.tensor(np.load(cube_dir+'dispersed_mask.npy')).float().to(device) #this needs changing also as I just made it by thresholding data.
    except:
        print('no mask found, creating one.')
        mask = undisp_cube > 0.05 * undisp_cube.max() #this is a hack.
        mask = np.transpose(mask[np.newaxis],(0,3,1,2))

        mask = torch.tensor(mask)
        dispersed_mask = torch.abs(fwd.fourier.method.disperser.disperse_all_orders(mask,kernel.to('cpu')))
        dispersed_mask[dispersed_mask<0.01] = 0
        dispersed_mask[dispersed_mask>0] = 1

        np.save(anadir+'dispersed_mask.npy',dispersed_mask.numpy())
        dispersed_mask = torch.tensor(dispersed_mask).to(device)

    ###normalize and send to cuda###
    undisp_cube = torch.tensor(normalize(undisp_cube)).float().permute(2,0,1).unsqueeze(0).to(device)
    spectras = torch.tensor(normalize(spectras)).float().to(device)

    return undisp_cube, dispersed_mask, spectras




def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



def downsample_signal(data, desired_channels, initial_range = [700,900], desired_range = [750,850], interp_axis = -1, interp_type='nearest'):
    '''
    TO BE RENAMED to DOWNSAMPLE_SIGNAL
    simplest possible example.
    desired_channels is desired channels.
    interp_axis is the axis to interpolate on.
    interp_type is the type of interpolation. Can be 'nearest'(fast) or 'average'(slow)
    '''

    if interp_axis==-1:
        interp_axis = len(data.shape)-1
    
    actual_bins = np.linspace(initial_range[0],initial_range[1],data.shape[interp_axis]) * 1e-9 

    desired_bins = np.linspace(desired_range[0],desired_range[1],desired_channels)*1e-9

    
    
    if interp_type == 'nearest':
        idx = np.zeros_like(desired_bins,dtype=int)
        for i in np.arange(len(desired_bins)):
            idx[i] = (np.abs(actual_bins - desired_bins[i])).argmin() #just find the closest one. (can replace with mean or something)
        newdata = np.take(data,idx,axis=interp_axis)

    elif interp_type == 'average':
        newshape = list(data.shape)
        newshape[interp_axis] = desired_channels
        newdata = np.zeros(newshape)

        dlambda = (desired_bins[1] - desired_bins[0])/2

        for i in np.arange(desired_channels):
            idx = np.where((actual_bins > desired_bins[i] - dlambda) * (actual_bins<desired_bins[i]+dlambda)) [0]
            indices = tuple([i if j==interp_axis else slice(None) for j in range(len(newshape))])
            newdata[indices] = np.mean(np.take(data,idx,axis=interp_axis),axis=interp_axis)

    return newdata

def load_cube(anadir):
    signalfft_center = np.load(anadir+'signalfft_padded_center.npy')
    signalfft_left = np.load(anadir+'signalfft_padded_left.npy')
    signalfft_right = np.load(anadir+'signalfft_padded_right.npy')
    undisp_cube = np.concatenate((signalfft_left,signalfft_center,signalfft_right),axis = 1) 
    return undisp_cube



def center_cubes(nograting_cube, grating_cube):
    '''
    given a cube without grating and one with grating, match the locations of the fundamental.
    '''

    sx,sy = nograting_cube.shape[2:]

    center_x, center_y = torch.sum(nograting_cube[0],dim=(0,2)).argmax(), torch.sum(nograting_cube[0],dim=(0,1)).argmax()

    rollx = (center_x-sx//2,center_y-sy//2)

    pointspot = sx//2  + rollx[0]   , sy//2+ rollx[1]

    size = 50 //2


    nograting_cube_rolled = torch.roll(nograting_cube, shifts=(-rollx[0],-rollx[1]), dims=(2, 3))
    grating_cube_rolled = torch.roll(grating_cube, shifts=(-rollx[0],-rollx[1]), dims=(2, 3)) # shift them both to the center of nograting_cube

    pointspot = sx//2    , sy//2


    grating_maxloc = torch.stack(torch.where(torch.sum(grating_cube_rolled[0,:,pointspot[0]-size:pointspot[0]+size+1,pointspot[1]-size:pointspot[1]+size+1].cpu().detach(),dim=0) == torch.sum(grating_cube_rolled[0,:,pointspot[0]-size:pointspot[0]+size+1,pointspot[1]-size:pointspot[1]+size+1].cpu().detach(),dim=0).max()))
    nograting_maxloc = torch.stack(torch.where(torch.sum(nograting_cube_rolled[0,:,pointspot[0]-size:pointspot[0]+size+1,pointspot[1]-size:pointspot[1]+size+1].cpu().detach(),dim=0) == torch.sum(nograting_cube_rolled[0,:,pointspot[0]-size:pointspot[0]+size+1,pointspot[1]-size:pointspot[1]+size+1].cpu().detach(),dim=0).max()))

    delta_pos = nograting_maxloc -  grating_maxloc  

    grating_cube_rolled = torch.roll(grating_cube_rolled, shifts=(delta_pos[0].numpy()[0],delta_pos[1].numpy()[0]), dims=(2, 3)) # shift the grating cube to the difference

    grating_cube = grating_cube_rolled
    nograting_cube = nograting_cube_rolled

    return nograting_cube, grating_cube