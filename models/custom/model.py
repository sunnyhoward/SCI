import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from forward.fourier.method import calc_psiT_g
from models.backbone.unet.modules import *
from models.backbone.unet.model import *
from models.custom.modules import *


class fourier_denoiser(nn.Module):
    """
    In this model we pass the measurement (9 copies) through a simple cnn, then we do the fourier deconvolution, before cropping and putting the output through a denoising UNet (w/ CoordGate).
    Input: mask - the predispersed mask-cube (bs,nc,nx,ny)
    """

    def __init__(self, mask, kernel, CoordGate=True, trainable_kernel=False):
        super().__init__()

        self.trainable_kernel = trainable_kernel
        self.kernel = kernel

        if trainable_kernel:
            
            locations = np.array([[[3,2039,1315,630,640,1320,630,1310],[10,2046,1420,730,740,1420,730,1410]], [[850,1450,8,2433,850,880,1440,1465],[1000,1590,16,2440,980,1010,1570,1595]]])
            locations = locations[...,:1]

            self.locations = locations

            variable_kernel = []
            for i in range(locations.shape[-1]):
                # self.kernel[0,:,locations[0,0,i]:locations[0,1,i],locations[1,0,i]:locations[1,1,i]] = nn.Parameter(kernel[0,:,locations[0,0,i]:locations[0,1,i],locations[1,0,i]:locations[1,1,i]]) 
                variable_kernel.append(nn.Parameter(kernel[0,:,locations[0,0,i]:locations[0,1,i],locations[1,0,i]:locations[1,1,i]]))
                # kernel_mask[0,:,locations[0,0,i]:locations[0,1,i],locations[1,0,i]:locations[1,1,i]] = 1 
            self.variable_kernel = variable_kernel
        
        self.mask = mask
        
        self.cropsize = 640//2

        if CoordGate:
            self.unet = CG_UNet(21,21, n_levels=5, init_size=[self.cropsize*2,self.cropsize*2])
        else:
            self.unet = UNet(21,21, n_levels=5)



    def data_term(self,x):
        kernel = self.fill_kernel()
        self.shift_info = {'kernel':kernel}
        x = calc_psiT_g(self.mask, x, self.shift_info) #(bs,nc,nx,ny)
        return x
    

    def fill_kernel(self):
        if self.trainable_kernel:
            kernel = torch.zeros_like(self.kernel)
            locations = self.locations
            for i in range(locations.shape[-1]):
                kernel[0,:,locations[0,0,i]:locations[0,1,i],locations[1,0,i]:locations[1,1,i]] = self.variable_kernel[i]
        else:
            kernel = self.kernel
        return kernel


    def crop(self,x):
        nx,ny = x.shape[2:]
        return x[...,nx//2 - self.cropsize : nx//2+self.cropsize,ny//2 - self.cropsize : ny//2+self.cropsize]

        

    def forward(self, x):
        '''
        Input: the measurement - (batch_size, nx, ny)
        '''

        #first id like to add some convolution to the original measurement (and to the kernel).
        
        x = self.data_term(x)   
        x = self.crop(x)
        x = self.unet(x)

        return x



class crop_denoiser(nn.Module):
    """
    In this model we pass the measurement (9 copies) through a simple cnn, then we do the fourier deconvolution, before cropping and putting the output through a denoising UNet (w/ CoordGate).
    Input: mask - the predispersed mask-cube (bs,nc,nx,ny)
    """

    def __init__(self, mask, CoordGate=True):
        super().__init__()

        self.mask = mask
        self.cropsize = 640//2

        if CoordGate:
            self.unet = CG_UNet(21,21, n_levels=5, init_size=[self.cropsize*2,self.cropsize*2])
        else:
            self.unet = UNet(21,21, n_levels=5)



    def crop(self,x):
        nx,ny = x.shape[2:]
        return x[...,nx//2 - self.cropsize : nx//2+self.cropsize,ny//2 - self.cropsize : ny//2+self.cropsize]

        

    def forward(self, x):
        '''
        Input: the measurement - (batch_size, nx, ny)
        '''

        #first id like to add some convolution to the original measurement (and to the kernel).
        x = self.mask * torch.tile(x.unsqueeze(1),(1,21,1,1))
        x = self.crop(x)
        x = self.unet(x)

        return x




class CG_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_levels, bilinear=False, BN=False, init_size = [640,640],device='cuda'):
        super(CG_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        clist = [64,128,256,512,1024]

        self.inc = (DoubleConv(n_channels, clist[0],BN=BN))

        size = torch.tensor(init_size)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.gates_down = nn.ModuleList()
        self.gates_up = nn.ModuleList()

        factor = 2 if bilinear else 1

        sizes = [torch.tensor([x,x]) for x in init_size[0] / 2**torch.arange(n_levels)] #fix for variable x and y

        for i in range(n_levels-1): 
            
            
            self.gates_down.append(CoordGate(3, 64, clist[i], size=sizes[i])) #the coordgate comes before the downsample

            if i!=n_levels-2:
                self.downs.append(Down(clist[i], clist[i+1],BN=BN))
            else:
                self.downs.append(Down(clist[i], clist[i+1]//factor,BN=BN))


            self.gates_up.append(CoordGate(3, 64, clist[n_levels-1 -i]//factor, size=sizes[n_levels-1 -i])) #coordgate comes before the upsample.
            self.ups.append(Up(clist[n_levels-1-i], clist[n_levels-1 -i-1]//factor, bilinear,BN=BN))
            size = torch.div(size, 2, rounding_mode='floor')



        self.final_gate = CoordGate(3, 64, 64, size=init_size)
        self.outc = (OutConv(clist[0], n_classes))




    def forward(self, x):
        
        x = self.inc(x)

        x_skips = []
        for i in range(len(self.downs)):
            x = self.gates_down[i](x)
            x_skips.append(x)
            x = self.downs[i](x)

        for i in range(len(self.ups)):
            x = self.gates_up[i](x)
            x = self.ups[i](x, x_skips[-i-1])

        x = self.final_gate(x)
        y = self.outc(x)
        return y