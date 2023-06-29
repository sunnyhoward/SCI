import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from forward.fourier.method import *
from models.backbone.unet.modules import *
from models.backbone.unet.model import *
from models.custom.modules import *


class FourierDenoiser(nn.Module):
    """
    In this model we do fourier deconvolution on measurement (9 copies), before cropping and putting the output through a denoising UNet (w/ CoordGate). Optionally we can also have the kernel be trainable.
    Input: mask - the predispersed mask-cube (bs,nc,nx,ny)
    """

    def __init__(self, mask, kernel, CoordGate=True, trainable_kernel=False, name=None):
        super().__init__()

        self.trainable_kernel = trainable_kernel
        self.kernel = kernel


        if trainable_kernel:
            
            self.kernel_learner = KernelLearner(self.kernel,  name='kernel_learner')

        
        self.relu = nn.ReLU()
        
        self.mask = mask
        
        self.wiener_noise = nn.Parameter(torch.tensor([1e-3]),requires_grad=True)

        self.cropsize = 640//2

        if CoordGate:
            self.unet = CG_UNet(21,21, n_levels=5, init_size=[self.cropsize*2,self.cropsize*2])
        else:
            self.unet = UNet(21,21, n_levels=5)

        self.name = f'FourierDenoiser_CG_{CoordGate}_trainkern_{trainable_kernel}' if name is None else name



    def forward(self, x):
        '''
        Input: the measurement - (batch_size, nx, ny)
        '''

        x = self.data_term(x)   
        x = self.crop(x)
        x = self.unet(x)

        return x


    def data_term(self,x):
        kernel = self.kernel_learner.fill_kernel() if self.trainable_kernel else self.kernel
        self.shift_info = {'kernel':kernel}
        x = calc_psiT_g(self.mask, x, self.shift_info, lamb=self.relu(self.wiener_noise)) #(bs,nc,nx,ny)
        return x
    

    def crop(self,x):
        nx,ny = x.shape[2:]
        return x[...,nx//2 - self.cropsize : nx//2+self.cropsize,ny//2 - self.cropsize : ny//2+self.cropsize]
     

    

class CG_UNet(nn.Module):
    '''
    UNet with CoordGate
    '''
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
    



class KernelLearner(nn.Module):
    """
    Here we make the kernel trainable. We do this by finding where the original kernel is non-zero, and then making all elements around the padding as trainable. 
    The forward call simply simulates the forward measurement.
    """

    def __init__(self, kernel , padding = 4, name=None):
        super().__init__()

        self.kernel = kernel

            
        locations = findclusters(kernel,padding) #searches for places where the kernel is not zero and creates a mask

        self.locations = nn.Parameter(locations,requires_grad=False)


        self.variable_kernel = nn.Parameter(kernel[0,locations], requires_grad=True)

        self.relu = nn.ReLU()  #nn.LeakyReLU()
        

        self.name = 'KernelLearner' if name is None else name


    def forward(self, x):
        '''
        Input: the cube (multiplied by spectra) - (batch_size, nc, nx, ny)
        '''

        #first id like to add some convolution to the original measurement (and to the kernel).

        kernel = self.fill_kernel()

        x = disperser.disperse_all_orders(x,kernel) #perform measurement

        return x



    def fill_kernel(self):
        kernel = torch.zeros_like(self.kernel)
        kernel[0,self.locations] = self.relu(self.variable_kernel)

        return kernel



class AffineTransformModel(nn.Module):
    '''
    A model that can apply seperate rotations and translations in certain regions. 
    '''
    def __init__(self, region, rot=20., transX=0., transY=0.):
        super().__init__()

        no_regions = len(region)

        self.rot_list = nn.Parameter(torch.deg2rad(torch.tensor([rot] * no_regions)))
        self.transX_list = nn.Parameter(torch.tensor([transX] * no_regions))
        self.transY_list = nn.Parameter(torch.tensor([transY] * no_regions))

        self.pos = region

        self.theta = nn.Parameter(torch.zeros(no_regions, 1, 2, 3), requires_grad = False)#.to(device)

        self.no_regions = no_regions

        self.grating_spectrum = nn.Parameter(torch.ones(no_regions, 21), requires_grad = True)


    def forward(self, x):
        transformed_x = x.clone()

        theta = self.fill_theta()

        for i in range(self.no_regions):
           
            grid = F.affine_grid(theta[i], x[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]].size(), align_corners=False)
            transformed_x[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]] = F.grid_sample(x[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]], grid, align_corners=False) * self.grating_spectrum[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return transformed_x
    

    def fill_theta(self):
        theta = torch.zeros_like(self.theta)

        for i in range(self.no_regions):
            rot_i = torch.cos(self.rot_list[i])  # Access the rotation parameter directly
            sin_i = torch.sin(self.rot_list[i])

            theta[i,:,0,0] = rot_i
            theta[i,:,0,1] = -sin_i
            theta[i,:,1,0] = sin_i
            theta[i,:,1,1] = rot_i
            theta[i,:,0,2] = self.transY_list[i]
            theta[i,:,1,2] = self.transX_list[i]

        return theta