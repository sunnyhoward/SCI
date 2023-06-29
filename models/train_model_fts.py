import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.parallel import DataParallel
import os, sys
main_dir = os.path.dirname(os.path.abspath(''))
sys.path.insert(0, main_dir)

import models.helper as hlp
from models.custom.model import *
import forward.fourier.method as fwd
from forward.fourier.kernel_creator import create_fourier_kernel

import gc
def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

gpuno = 3
device = f'cuda:{gpuno}'

kernel = torch.load('kernel.pt',map_location=device)

fts_dir = '20230628_2gratings_data_1500us_talbot_0'

undisp_cube, mask, spectras = hlp.create_bs_data(desired_channels=21,kernel=kernel, fts_dir = fts_dir, cube_dir ='20230628_analysis_300us_talbot_0',  interp_type='average',device=device) #here we'll work with synthetic data. 

batch_size = 2
exist = False
# torch.cuda.empty_cache()

dataset = hlp.FTSDataset(undispersed_cube=undisp_cube,dir=fts_dir, spectra = spectras) #chose the dir here idiot

allindexes = np.arange(len(dataset))
np.random.shuffle(allindexes)
tr_indexes = allindexes[:int(0.8*len(allindexes))]
v_indexes = allindexes[int(0.8*len(allindexes)):] 

# Create a data loader for batch processing
tr_loader = hlp.CustomDataLoader(dataset, tr_indexes, batch_size=batch_size, shuffle=True)
v_loader = hlp.CustomDataLoader(dataset, v_indexes, batch_size=batch_size, shuffle=True)

model = FourierDenoiser(mask=mask,kernel=kernel,CoordGate=False,name='FTS_unet').to(device)
#model = DataParallel(model,device_ids=[1,2,3]).to(0)

if exist:
    model.load_state_dict(torch.load('FTS_unet')); #trained on all data.
    lr = 5e-5
else:
    lr = 5e-4
    
epochs = 50



optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.MSELoss()


history = hlp.train(model, optimizer, loss, tr_loader, v_loader, epochs=epochs, device=device)
with torch.cuda.device(device):
    torch.cuda.empty_cache()



torch.save(model.state_dict(), model.name)
