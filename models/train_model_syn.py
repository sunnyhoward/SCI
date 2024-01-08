import os
import sys
main_dir = os.path.dirname(os.path.abspath(''))
sys.path.insert(0, main_dir)

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.parallel import DataParallel



import gc

import forward.fourier.method as fwd
import models.helper as hlp
from forward.fourier.kernel_creator import create_fourier_kernel
from models.custom.model import *


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if torch.cuda.is_available():
    torch.cuda.set_device(2) # Select 5th GPU
    print(torch.cuda.device_count(), "GPU(s) available:")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
else:
    print("No GPUs available.")

gpuno = 3
device = f'cuda:{gpuno}'
desired_range = [735,865]
desired_channels = 50


kernel = torch.load('final_kernel.pt',map_location=device).requires_grad_(False)

undisp_cube = torch.tensor(np.load('/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/20230906_1grating_mask_analysis_exptime_6250_us/gtcube.npy')).to(device)
mask = torch.tensor(np.load('/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/20230906_1grating_mask_analysis_exptime_6250_us/mask.npy')).to(device)[:1]


nc = undisp_cube.shape[1]

datadir = '/project/agdoepp/Experiment/Hyperspectral_Calibration_FTS/20230906_1grating_mask_data_exptime_6250_us/' 
    
orig_spectras = np.load(datadir+'spectra.npy') 
initial_bins = np.linspace(634.69, 1124.5, orig_spectras.shape[0])

spectras, final_bins = hlp.fix_spectra(orig_spectras, initial_bins, desired_range = desired_range)
spectras[(final_bins<745) | (final_bins>855)] = 0

spectras = hlp.downsample_signal(spectras, desired_channels, final_bins * 1e-9, desired_range = desired_range, interp_axis = 0, interp_type='average') #interpolate to 750-850nm
spectras = torch.tensor(spectras).float().to(device)
cutoff = [450,950]
spectras = spectras[:,cutoff[0]:cutoff[1]]


batch_size = 4
torch.cuda.empty_cache()

cropsize = [500, 640]

dataset = hlp.SyntheticDataset(undispersed_cube=undisp_cube,shift_info={'kernel':kernel}, spectra = spectras, crop=cropsize, random_shifts=False)

allindexes = np.arange(len(dataset));  np.random.shuffle(allindexes)

tr_split = int((0.8*len(allindexes)) // batch_size * batch_size)
v_split = -int((0.2*len(allindexes)) // batch_size * batch_size)

tr_indexes = allindexes[:tr_split]
v_indexes = allindexes[v_split:] 

# Create a data loader for batch processing
tr_loader = hlp.CustomDataLoader(dataset, tr_indexes, batch_size=batch_size, shuffle=True)
v_loader = hlp.CustomDataLoader(dataset, v_indexes, batch_size=batch_size, shuffle=True)

model = FourierDenoiser(kernel=kernel, mask=mask, CoordGate=False, channels=nc, cropsize=cropsize).to(device)
# model = DataParallel(model,device_ids=[0,1]).to(device)


lr = 1e-4
epochs = 100


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.MSELoss()


history = hlp.train(model, optimizer, loss, tr_loader, v_loader, epochs=epochs, device=device)

torch.cuda.empty_cache()

torch.save(model.state_dict(), 'syn')
