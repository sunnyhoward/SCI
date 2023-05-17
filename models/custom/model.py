import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# import os
# import sys
# main_dir = os.path.dirname(os.path.abspath('../'))
# sys.path.insert(0, main_dir)

from forward.fourier.method import calc_psiT_g
from models.backbone.unet.modules import *
from models.backbone.unet.model import *
from models.custom.modules import *



class fourier_denoiser(nn.Module):
    """
    In this model we pass the measurement (9 copies) through a simple cnn, then we do the fourier deconvolution, before cropping and putting the output through a denoising UNet (w/ CoordGate).
    Input: mask - the predispersed mask-cube (bs,nc,nx,ny)
    """

    def __init__(self, mask, kernel, CoordGate=True, ):
        super().__init__()

        
        self.kernel = kernel #nn.Parameter(kernel, requires_grad =True) #maybe not necessary    
        self.shift_info = {'kernel':self.kernel}
    
        self.mask = mask
        
        self.cropsize = 640//2

        if CoordGate:
            self.unet = CG_UNet(21,21, n_levels=5, init_size=[self.cropsize*2,self.cropsize*2])
        else:
            self.unet = UNet(21,21, n_levels=5)



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




class CG_UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_levels, bilinear=False, BN=False, init_size = [640,640]):
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
            size = size//2



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