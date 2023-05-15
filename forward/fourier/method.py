import numpy as np
import torch


def calc_psi_z(psi,z,kernel):
    '''
    Simulate the forward sensing process

            Parameters:
                    psi (array): The mask
                    z (array): the cube (bs,nc,nx,ny)
                    kernel (dict): The fourier kernel to do dispersion.

            Returns:
                    (array): dispersed, integrated cube
    '''

    z_disp = disperser.disperse_all_orders(z,kernel)

    return torch.sum(psi*z_disp,axis=1)


def calc_psiT_g(psi,g,kernel):
    '''
    Transpose of the sensing process.

            Parameters:
                    psi (array): The mask
                    g (array): the sensor image
                    kernel (dict): The fourier kernel to do dispersion.

            Returns:
                    (array): undispersed, unintegrated cube
    '''
     
    nc = psi.shape[1]

    g = g / (torch.sum(psi,axis=1)) #will break if psi has rows of 0s

    cube = psi * torch.tile(g[:,:,:,torch.newaxis],(1,nc,1,1))

    cube_disp = disperser.undisperse_all_orders(cube,kernel)

    return cube_disp


class disperser:
    @staticmethod
    def disperse_all_orders(cube, kernel):
        f_cube = torch.fft.fft2(cube)
        f_kernel = torch.fft.fft2(kernel)
        f_cube_disp = f_cube * f_kernel
        cube_disp = torch.fft.ifft2(f_cube_disp)
        return cube_disp
    

    @staticmethod
    def undisperse_all_orders(cube, kernel):
        f_cube = torch.fft.fft2(cube)
        f_kernel = torch.fft.fft2(kernel)
        f_cube_undisp = f_cube / f_kernel
        cube_undisp = torch.fft.ifft2(f_cube_undisp)
        return cube_undisp
        

