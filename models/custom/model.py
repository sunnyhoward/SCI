import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# import os
# import sys
# main_dir = os.path.dirname(os.path.abspath('../'))
# sys.path.insert(0, main_dir)

from forward.fourier.method import calc_psiT_g
from models.backbone.unet.model import UNet



class fourier_denoiser(nn.Module):
    """
    In this model we pass the input first through a simple cnn, then we do the fourier deconvolution, before cropping and putting the output through a denoising UNet.
    Input: mask - the predispersed mask-cube (bs,nc,nx,ny)
    """

    def __init__(self, mask, kernel):
        super().__init__()

        
        self.kernel = kernel #nn.Parameter(kernel, requires_grad =True) #maybe not necessary    
        self.shift_info = {'kernel':self.kernel}
    
        self.mask = mask
        
        self.cropsize = 640//2

        self.unet = UNet(21,21)


    def data_term(self,x):

        x = calc_psiT_g(self.mask, x, self.shift_info) #(bs,nc,nx,ny)

        return x

    def crop(self,x):
        nx,ny = x.shape[2:]
        return x[...,nx//2 - self.cropsize : nx//2+self.cropsize,ny//2 - self.cropsize : ny//2+self.cropsize]

        

    def forward(self, x):
        '''
        Input: the measurement - (batch_size, nx, ny)
        '''
        
        x = self.data_term(x)   
        x = self.crop(x)
        x = self.unet(x)

        return x

