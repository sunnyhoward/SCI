import torch
from torch.utils.data import Dataset
import h5py
import forward as fwd
import time
import numpy as np

def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='gpu'):
    '''
    The function to train a model.

    @TODO: Add the ability to redirect the print statements to the model_results.txt file.
    '''

    print('train called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

    total_batches = int(len(train_dl.indexes)/train_dl.batch_size)

    start_time_sec = time.time()

    for epoch in range(1, epochs+1):
        # --- TRAIN AND EVALUATE ON TRAINING SET -----------------------------
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


        print(f'Epoch {epoch}/{epochs}: ({total_batches}/{total_batches}), train loss: {train_loss:5.5g}, val_loss = {val_loss:5.5g}')


        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    end_time_sec       = time.time()
    total_time_sec     = end_time_sec - start_time_sec
    time_per_epoch_sec = total_time_sec / epochs
    print()
    print('Time total:     %5.2f sec' % (total_time_sec))
    print('Time per epoch: %5.2f sec' % (time_per_epoch_sec))

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

            # batch_data = [list(self.dataset[idx]) for idx in batch_indices]
            # batch = list(zip(*batch_data))
            # x,y = torch.cat(batch[0]),torch.cat(batch[1])

            x,y = self.dataset[batch_indices]
            yield x,y

    def __len__(self):
        return len(self.dataset) // self.batch_size



class SyntheticDataset(Dataset):
    '''
    Generate data for training by taking an undispersed cube, dispersing it and applying spectral modulation.

            Parameters:
                    undispersed_cube (array)
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

        x = self.data * self.spectra[:,index].permute(1,0).unsqueeze(-1).unsqueeze(-1) # Convert data to a PyTorch tensor

        y = self.sensing_function(torch.ones_like(x),x,shift_info=self.shift_info)  # Convert labels to a PyTorch tensor
        
        # y = y.unsqueeze(1)
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
    



def create_bs_data(desired_channels,interp_type='nearest'):
    '''
    This is ugly so put it in a function. 

    Calling this function will yield the undispersed cube, the dispersed mask and the FTS spectra.
    '''
    ###load the cube###
    anadir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/20230508_mask_analysis_talbot_0_300us/'
    datadir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/20230508_mask_data_talbot_0_300us'
    signalfft_center = np.load(anadir+'signalfft_padded_center.npy')
    signalfft_left = np.load(anadir+'signalfft_padded_left.npy')
    signalfft_right = np.load(anadir+'signalfft_padded_right.npy')
    undisp_cube = np.concatenate((signalfft_left,signalfft_center,signalfft_right),axis = 1) 
    
    ###load the mask and FTS spectra###
    spectras = np.load(datadir+'/spectra.npy') 
    mask = torch.tensor(np.load(anadir+'/dispersed_mask.npy')).float().to('cuda') #this needs changing also as I just made it by thresholding data.

    ###interpolate to desired channels###
    undisp_cube = interpolate_signal(undisp_cube, desired_channels, index_axis = 2, interp_type=interp_type) #interpolate to 750-850nm
    spectras = interpolate_signal(spectras, desired_channels, index_axis = 0, interp_type=interp_type) #interpolate to 750-850nm

    ###normalize and send to cuda###
    undisp_cube = torch.tensor(normalize(undisp_cube)).float().permute(2,0,1).unsqueeze(0).to('cuda')
    spectras = torch.tensor(normalize(spectras)).float().to('cuda')

    return undisp_cube, mask, spectras




def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))



def interpolate_signal(data, desired_channels, index_axis, interp_type='nearest'):
    '''
    simplest possible example.
    desired_channels is desired channels.
    index_axis is the axis to interpolate on.
    interp_type is the type of interpolation. Can be 'nearest'(fast) or 'average'(slow)
    '''

    desired_bins = np.linspace(750,851,desired_channels)*1e-9

    actual_bins = np.linspace(700,900,data.shape[index_axis]) * 1e-9 #assuming we are from 700 to 900

    
    if interp_type == 'nearest':
        idx = np.zeros_like(desired_bins,dtype=int)
        for i in np.arange(len(desired_bins)):
            idx[i] = (np.abs(actual_bins - desired_bins[i])).argmin() #just find the closest one. (can replace with mean or something)
        newdata = np.take(data,idx,axis=index_axis)

    elif interp_type == 'average':
        newshape = list(data.shape)
        newshape[index_axis] = desired_channels
        newdata = np.zeros(newshape)

        dlambda = (desired_bins[1] - desired_bins[0])/2

        for i in np.arange(desired_channels):
            idx = np.where((actual_bins > desired_bins[i] - dlambda) * (actual_bins<desired_bins[i]+dlambda)) [0]
            indices = tuple([i if j==index_axis else slice(None) for j in range(len(newshape))])
            newdata[indices] = np.mean(np.take(data,idx,axis=index_axis),axis=index_axis)

    return newdata
