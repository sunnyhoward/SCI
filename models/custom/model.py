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

    def __init__(self, kernel, channels, mask=None, CoordGate=True, trainable_kernel=False, cropsize = [640,640], name=None):
        super().__init__()

        self.trainable_kernel = trainable_kernel
        self.kernel = nn.Parameter(kernel, requires_grad=False)


        if trainable_kernel:
            
            self.kernel_learner = KernelLearner(self.kernel,  name='kernel_learner')
        
        self.mask = mask  #if we are adjusting the kernel we need to make a new mask each time
        
        self.relu = nn.ReLU()
        
        
        
        self.wiener_noise = nn.Parameter(torch.tensor([1e-3]),requires_grad=True)

        self.cropsize = cropsize

        if CoordGate:
            self.unet = CG_UNet(channels,channels, n_levels=5, init_size=self.cropsize)
        else:
            self.unet = UNet(channels,channels, n_levels=5)

        self.name = f'FourierDenoiser_CG_{CoordGate}_trainkern_{trainable_kernel}' if name is None else name



    def forward(self, x):
        '''
        Input: the measurement - (batch_size, nx, ny)
        '''


        kernel = self.kernel_learner.fill_kernel() if self.trainable_kernel else self.kernel
        mask = self.make_mask(x,kernel, self.cropsize) if self.mask == None else self.mask

        x = self.data_term(x, kernel, mask)   

        x = self.crop(x, self.cropsize)

        x = self.unet(x)

        return x


    
    def data_term(self,x,kernel,mask):
        
        self.shift_info = {'kernel':kernel}
        x = calc_psiT_g(mask, x, self.shift_info, lamb=self.relu(self.wiener_noise)) #(bs,nc,nx,ny)
        return x
    

    @staticmethod
    def make_mask(x,kernel,cropsize):
        #x is measurement

        nx,ny = x.shape[-2:]
        mask = torch.zeros_like(x)
        cropx = cropsize[0]//2
        cropy = cropsize[1]//2

        x_cropped = FourierDenoiser.crop(x, cropsize)
        mask[...,nx//2 - cropx : nx//2+cropx,ny//2 - cropy : ny//2+cropy] = x_cropped > (0.005 * x_cropped.max()) #this is a hack.
        mask = mask.unsqueeze(1).tile(1,kernel.shape[1],1,1)

        dispersed_mask = disperser.disperse_all_orders(mask,kernel)
        bla = torch.stack([dispersed_mask[:,i] > dispersed_mask[:,i].mean() * 0.005 for i in range(dispersed_mask.shape[1])] , dim=1)
        dispersed_mask[bla] = 1
        dispersed_mask[dispersed_mask != 1] = 0
        return dispersed_mask
    
    
    @staticmethod
    def crop(x,cropsize):
        nx,ny = x.shape[-2:]
        cropx = cropsize[0]//2
        cropy = cropsize[1]//2

        return x[...,nx//2 - cropx : nx//2+cropx,ny//2 - cropy : ny//2+cropy]
     

    

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
            
            
            self.gates_down.append(CoordGate(encoding_layers=3, enc_channels=64, out_channels=clist[i], size=sizes[i])) #the coordgate comes before the downsample

            if i!=n_levels-2:
                self.downs.append(Down(clist[i], clist[i+1],BN=BN))
            else:
                self.downs.append(Down(clist[i], clist[i+1]//factor,BN=BN))


            self.gates_up.append(CoordGate(encoding_layers=3, enc_channels=64, out_channels=clist[n_levels-1 -i]//factor, size=sizes[n_levels-1 -i])) #coordgate comes before the upsample.
            self.ups.append(Up(clist[n_levels-1-i], clist[n_levels-1 -i-1]//factor, bilinear,BN=BN))
            size = torch.div(size, 2, rounding_mode='floor')



        self.final_gate = CoordGate(encoding_layers=3, enc_channels=64, out_channels=64, size=init_size)
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
    


class CG_convolution_layer(nn.Module):
    '''
    Single convolution layer with CoordGate

    if three_d is true, they will share convolutions but not the coordgate.
    '''
    def __init__(self, n_channels_in, n_channels_out, n_channels_CG, kernelsize, init_size = [640,640],
                 locally_connected=False,CG_type='pos', three_d=False,**kwargs):
        super(CG_convolution_layer, self).__init__()
        self.kernelsize = kernelsize

        self.conv = nn.Conv2d(n_channels_in,n_channels_CG,kernelsize,padding=kernelsize//2)
        if locally_connected:
            self.create_locally_connected_conv(kernelsize)

        if three_d:
            channels = kwargs['channels']
            self.CG = torch.nn.ModuleList([CoordGate(enc_channels=n_channels_CG, out_channels=n_channels_out, size=init_size, enctype=CG_type, **kwargs) for i in range(channels)])
        else:
            self.CG = CoordGate(enc_channels=n_channels_CG, out_channels=n_channels_out, size=init_size, enctype=CG_type, **kwargs)

        self.three_d = three_d


    def forward(self, x):
        if self.three_d:
            x = torch.concat([self.CG[i](self.conv(x[:,i])) for i in range(x.shape[1])],dim=1)
        else:
            x = self.CG(self.conv(x))
        return x
    
    def create_locally_connected_conv(self,kernelsize):
        locally_connected_kernel = torch.zeros((kernelsize**2,1,kernelsize,kernelsize))

        for i in range(kernelsize):
            for j in range(kernelsize):
                locally_connected_kernel[i*kernelsize+j,0,i,j] = 1
        self.conv.weight = nn.Parameter(locally_connected_kernel,requires_grad=False)




class KernelLearner(nn.Module):
    """
    Here we make the kernel trainable. We do this by finding where the original kernel is non-zero, and then making all elements around the padding as trainable. 
    The forward call simply simulates the forward measurement.
    """

    def __init__(self, kernel , padding = 4, name=None):
        super().__init__()

        self.kernel = kernel

        self.lamb = nn.Parameter(torch.tensor([1e-4]),requires_grad=True)
            
        locations = findclusters(kernel,padding) #searches for places where the kernel is not zero and creates a mask
        self.locations = nn.Parameter(locations,requires_grad=False)
        self.initialise_variable_kernel()

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
    

    def forward_inverse(self, x):

        kernel = self.fill_kernel()
        
        x = disperser.undisperse_all_orders(x,kernel,wiener=True,lamb=self.relu(self.lamb))
        
        return x    
    

    def initialise_variable_kernel(self):
        self.variable_kernel = nn.Parameter(self.kernel[0,self.locations] + 1e-10, requires_grad=True)


    def fill_kernel(self):
        # kernel = torch.zeros_like(self.kernel)
        kernel = self.kernel.clone()
        kernel[0,self.locations] = self.relu(self.variable_kernel)
        return kernel




class GratingModulationLearner(nn.Module):
    """
    Here we make the kernel trainable. We do this by finding where the original kernel is non-zero, and then making all elements around the padding as trainable. 
    The forward call simply simulates the forward measurement.
    """

    def __init__(self, kernel , regions,  name=None):
        super().__init__()

        self.kernel = kernel


        self.pos = regions
            
        self.relu = nn.ReLU()  #nn.LeakyReLU()

        self.name = 'GratingModulationLearner' if name is None else name

        channels = kernel.shape[1]

        self.grating_spectrum = nn.Parameter(torch.ones(len(regions), channels), requires_grad = True)


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
        
        for i in range(len(self.pos)):
            kernel[0,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]] = self.relu(self.grating_spectrum[i]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1) * self.kernel[0,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]]

        # kernel = self.kernel * self.relu(self.factor)
   

        return kernel




class AffineTransformModel(nn.Module):
    '''
    A model that can apply seperate rotations and translations in certain regions. 
    '''
    def __init__(self, region, rot=20., transX=0., transY=0., scale=True, no_channels = 41, grating = True):
        super().__init__()

        no_regions = len(region)

        self.transX_list = nn.Parameter(torch.tensor([transX] * no_regions))
        self.transY_list = nn.Parameter(torch.tensor([transY] * no_regions))
        if scale:
            self.scaleX_list = nn.Parameter(torch.stack([torch.tensor([1., 0.]) for i in range(no_regions)]))
            self.scaleY_list = nn.Parameter(torch.stack([torch.tensor([0., 1.]) for i in range(no_regions)]))
        else: self.rot_list = nn.Parameter(torch.deg2rad(torch.tensor([rot] * no_regions)))


        self.pos = region

        self.theta = nn.Parameter(torch.zeros(no_regions, 1, 2, 3), requires_grad = False)#.to(device)

        self.no_regions = no_regions

        self.scale=scale
        self.grating = grating
        if grating:
            self.grating_spectrum = nn.Parameter(torch.ones(no_regions, no_channels), requires_grad = True)
        self.relu = nn.ReLU()


    def forward(self, x):
        transformed_x = x.clone() #kernel

        theta = self.fill_theta()

        for i in range(self.no_regions):

            grid = F.affine_grid(theta[i], x[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]].size(), align_corners=False)
            if self.grating:
                transformed_x[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]] = F.grid_sample(x[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]], grid, align_corners=False) * self.relu(self.grating_spectrum[i]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            else:
                transformed_x[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]] = F.grid_sample(x[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]], grid, align_corners=False) 

        return transformed_x


    def fill_theta(self):
        theta = torch.zeros_like(self.theta)

        for i in range(self.no_regions):
                        
            if self.scale:
                theta[i,:,0,:2] = self.scaleX_list[i]
                theta[i,:,1,:2] = self.scaleY_list[i]
            else:
                rot_i = torch.cos(self.rot_list[i])  # Access the rotation parameter directly
                sin_i = torch.sin(self.rot_list[i])
                theta[i,:,0,0] = rot_i
                theta[i,:,0,1] = -sin_i
                theta[i,:,1,0] = sin_i
                theta[i,:,1,1] = rot_i
            
            # theta[i,:,0,0] = 1
            # theta[i,:,1,1] = 1

            theta[i,:,0,2] = self.transY_list[i]
            theta[i,:,1,2] = self.transX_list[i]

        return theta
    
    def set_theta(self, theta ):
        
        self.transX_list.requires_grad = False
        self.transY_list.requires_grad = False
        self.scaleX_list.requires_grad = False
        self.scaleY_list.requires_grad = False

        self.transX_list = nn.Parameter(torch.tensor(theta[:,0,2]))
        self.transY_list = nn.Parameter(torch.tensor(theta[:,1,2]))
        self.scaleX_list = nn.Parameter(torch.tensor(theta[:,0,:2]))
        self.scaleY_list = nn.Parameter(torch.tensor(theta[:,1,:2]))

        self.transX_list.requires_grad = True
        self.transY_list.requires_grad = True
        self.scaleX_list.requires_grad = True
        self.scaleY_list.requires_grad = True


    def fit_angles(self, truth, init_guess, funda_idx = 4, verbose = False):
        '''
        Find the angles between the truth and initguess to match the kernel rot angles.
        '''
        
        if self.scale:
            self.scaleX_list.requires_grad = False
            self.scaleY_list.requires_grad = False
        else:
            self.rot_list.requires_grad = False
        
        for i in range(len(self.pos)):

            image_g = torch.sum(init_guess[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]],dim=(0,1)).cpu().detach()
            image_t = torch.sum(truth[:,:,self.pos[i,0,0]:self.pos[i,0,1],self.pos[i,1,0]:self.pos[i,1,1]],dim=(0,1)).cpu().detach()

            angle_g = find_angle(image_g)
            angle_t = find_angle(image_t)
            if verbose: print(torch.rad2deg(angle_g),torch.rad2deg(angle_t))


            if self.scale:
                self.scaleX_list[i] = torch.tensor([torch.cos(angle_t- angle_g), -torch.sin(angle_t- angle_g)])
                self.scaleY_list[i] = torch.tensor([torch.sin(angle_t- angle_g), torch.cos(angle_t- angle_g)])
            else:
                self.rot_list[i] = torch.tensor(angle_t - angle_g)


        if self.scale:
            self.scaleX_list[funda_idx] = torch.tensor([1, 0])
            self.scaleY_list[funda_idx] = torch.tensor([0, 1])
            self.scaleX_list.requires_grad = True
            self.scaleY_list.requires_grad = True
        else:
            
            self.rot_list[funda_idx] = 0 # this should be the fundamental.
            self.rot_list.requires_grad = True

    
    def fit_translation(self, truth, init_guess, intensity_factor = 1,  verbose = False):
        '''
        given a true cube and a pred cube, find difference in CoMs for each spectral channel along with std. Recommend using this one after fit_angles.
        '''
        store = torch.zeros((len(self.pos),2,2)) #region, mean/std, x/y

        self.transX_list.requires_grad = False
        self.transY_list.requires_grad = False

        lambda_range = [int(truth.shape[1]//3), int(truth.shape[1]*2//3)]

        # first find the differences in CoMs
        for i in range(len(self.pos)):
            diff = []

            for l in range(lambda_range[0],lambda_range[1]): 
                guess = CenterOfMassLoss.calculate_center_of_mass(init_guess[:,l],region = self.pos[i], intensity_factor = intensity_factor)
                true = CenterOfMassLoss.calculate_center_of_mass(truth[:,l],region = self.pos[i], intensity_factor = intensity_factor)

                diff.append(true - guess)

            diff = torch.stack(diff)

            store[i,0] = torch.mean(diff,dim=0)
            store[i,1] = torch.std(diff,dim=0)

            #put the relative translation in the right place
            self.transX_list[i] = -store[i,0,0].detach() / (self.pos[i,0,1] - self.pos[i,0,0])*2
            self.transY_list[i] = -store[i,0,1].detach() / (self.pos[i,1,1] - self.pos[i,1,0])*2
        
        self.transX_list.requires_grad = True
        self.transY_list.requires_grad = True

        if verbose: return store


    
    

class CubeToFTSModel(nn.Module):
    """
    Assume the cube has the spots in the right places with the right profile (as they should be considering )
    """

    def __init__(self, kernel , padding = 4, name=None):
        super().__init__()

        self.kernel = kernel
        self.type = type

        locations = findclusters(kernel,padding) #searches for places where the kernel is not zero and creates a mask

        self.locations = nn.Parameter(locations,requires_grad=False)


        self.variable_kernel = nn.Parameter(kernel[0,locations]+1e-11, requires_grad=True)

        self.relu = nn.ReLU()  #nn.LeakyReLU()
            

        self.name = 'KernelLearner' if name is None else name


    def forward(self, x):
        '''
        Input: the cube (multiplied by spectra) - (batch_size, nc, nx, ny)
        '''

        #first id like to add some convolution to the original measurement (and to the kernel).

        kernel = self.fill_kernel()

        x = calc_psi_z(torch.ones_like(x),x,shift_info={'kernel':kernel})

        return x



    def fill_kernel(self):
        kernel = torch.zeros_like(self.kernel)

        kernel[0,self.locations] = self.relu(self.variable_kernel)

        return kernel