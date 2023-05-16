import torch
from torch.utils.data import Dataset
import h5py
import forward as fwd
import time


def train(model, optimizer, loss_fn, train_dl, val_dl, epochs=100, device='gpu'):

    print('train() called: model=%s, opt=%s(lr=%f), epochs=%d, device=%s\n' % \
          (type(model).__name__, type(optimizer).__name__,
           optimizer.param_groups[0]['lr'], epochs, device))

    history = {} # Collects per-epoch loss and acc like Keras' fit().
    history['loss'] = []
    history['val_loss'] = []
    history['acc'] = []
    history['val_acc'] = []

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

            print('Epoch%3d/%3d: (%3d/%3d), train loss: %5.5g' % \
                (epoch, epochs,n, len(train_dl.indexes)/train_dl.batch_size, train_loss/num_train_examples),end='\r')

            # print('Batch %3d/%3d, loss: %5.5g'  % \
            #     (n, len(train_dl.indexes)/train_dl.batch_size, train_loss/num_train_examples),end='\r')
        
        # print('Batch %3d/%3d, loss: %5.5g'  % \
        #         (n, len(train_dl.indexes)/train_dl.batch_size, train_loss/num_train_examples))

        train_loss  = train_loss / len(train_dl.indexes)


        # --- EVALUATE ON VALIDATION SET -------------------------------------
        model.eval()
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


        # if epoch == 1 or epoch % 10 == 0:
        print('Epoch %3d/%3d: (%3d/%3d), train loss: %5.5g, val loss: %5.5g' % \
                (epoch, epochs,len(train_dl.indexes)/train_dl.batch_size, len(train_dl.indexes)/train_dl.batch_size, train_loss, val_loss))

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

    def __iter__(self):

        indices = self.indexes[torch.randperm(len(self.indexes))] if self.shuffle else self.indexes

        for batch_start in range(0, len(indices), self.batch_size):
            batch_indices = indices[batch_start:batch_start + self.batch_size]

            batch_data = [list(self.dataset[idx]) for idx in batch_indices]
            batch = list(zip(*batch_data))
            x,y = torch.cat(batch[0]),torch.cat(batch[1])
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

        self.data = torch.tensor(undispersed_cube)
        self.shift_info = shift_info
        self.spectra = torch.tensor(spectra)
        self.crop = crop    
        self.sensing_function = fwd.fourier.method.calc_psi_z if method == 'fourier' else fwd.rolling.method.calc_psi_z


    def __len__(self):
        return len(self.spectra[0])

    def __getitem__(self, index):
        
        x = self.data.permute(0,2,3,1) * self.spectra[:,index] # Convert data to a PyTorch tensor

        x = x.permute(0,3,1,2)

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
    


