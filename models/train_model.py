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
   # torch.cuda.set_device(2) # Select 5th GPU
    print(torch.cuda.device_count(), "GPU(s) available:")
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))
else:
    print("No GPUs available.")


undisp_cube, mask, spectras = hlp.create_bs_data(desired_channels=21,interp_type='average') #here we'll work with synthetic data. 


batch_size = 2
torch.cuda.empty_cache()


kernel = create_fourier_kernel().unsqueeze(0).permute(0,3,1,2).to('cuda')

dataset = hlp.SyntheticDataset(undispersed_cube=undisp_cube,shift_info={'kernel':kernel}, spectra = spectras)

allindexes = np.arange(len(dataset))
np.random.shuffle(allindexes)
tr_indexes = allindexes[:int(0.8*len(allindexes))]
v_indexes = allindexes[int(0.8*len(allindexes)):] 

# Create a data loader for batch processing
tr_loader = hlp.CustomDataLoader(dataset, tr_indexes, batch_size=batch_size, shuffle=True)
v_loader = hlp.CustomDataLoader(dataset, v_indexes, batch_size=batch_size, shuffle=True)

model = fourier_denoiser(mask=mask, kernel=kernel).to('cuda')
#model = DataParallel(model,device_ids=[1,2,3]).to(0)


lr = 5e-4
epochs = 50


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = torch.nn.MSELoss()


history = hlp.train(model, optimizer, loss, tr_loader, v_loader, epochs=epochs, device=0)

torch.cuda.empty_cache()

torch.save(model.state_dict(), 'fourier_denoiser_b')
