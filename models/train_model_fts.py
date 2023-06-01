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

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if torch.cuda.is_available():
    print(torch.cuda.device_count(), "GPU(s) available:")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
else:
    print("No GPUs available.")

gpuno = 4
device = f'cuda:{gpuno}'


undisp_cube, mask, spectras = hlp.create_bs_data(desired_channels=21, dir = '20230522_mask_2gratings_data_talbot_0_15000us',interp_type='average',device=device) #here we'll work with synthetic data. 
blamask = torch.zeros_like(undisp_cube)
sx,sy = undisp_cube.shape[2:]

blamask[...,sx//2-320:sx//2+320,sy//2-320:sy//2+320] = 1
undisp_cube = undisp_cube * blamask

del blamask
print(undisp_cube.shape, mask.shape, spectras.shape)

batch_size = 1
# torch.cuda.empty_cache()


kernel = create_fourier_kernel().unsqueeze(0).permute(0,3,1,2).to(device)

dataset = hlp.FTSDataset(undispersed_cube=undisp_cube, spectra = spectras)

allindexes = np.arange(len(dataset))
np.random.shuffle(allindexes)
tr_indexes = allindexes[:int(0.8*len(allindexes))]
v_indexes = allindexes[int(0.8*len(allindexes)):] 

# Create a data loader for batch processing
tr_loader = hlp.CustomDataLoader(dataset, tr_indexes, batch_size=batch_size, shuffle=True)
v_loader = hlp.CustomDataLoader(dataset, v_indexes, batch_size=batch_size, shuffle=True)

model = fourier_denoiser(mask=mask,kernel=kernel,CoordGate=False).to(device)
#model = DataParallel(model,device_ids=[1,2,3]).to(0)


lr = 5e-4
epochs = 30


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.MSELoss()


history = hlp.train(model, optimizer, loss, tr_loader, v_loader, epochs=epochs, device=device)
with torch.cuda.device(device):
    torch.cuda.empty_cache()

torch.save(model.state_dict(), 'unet_fts')
