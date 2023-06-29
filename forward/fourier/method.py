import numpy as np
import torch


def calc_psi_z(psi,z,shift_info):
    '''
    Simulate the forward sensing process

            Parameters:
                    psi (array): The mask
                    z (array): the cube (bs,nc,nx,ny)
                    shift_info (dict): The fourier kernel to do dispersion.

            Returns:
                    (array): dispersed, integrated cube
    '''
    kernel = shift_info['kernel']
    z_disp = disperser.disperse_all_orders(z,kernel)

    return torch.sum(psi*z_disp,axis=1)


def calc_psiT_g(psi, g, shift_info,  wiener=True, lamb=0.1, normalize=False): #@TODO: Normalize is broken
    '''
    Transpose of the sensing process.

            Parameters:
                    psi (array): The mask, carrying the same dispersion as the measurement (bs,nc,nx,ny)
                    g (array): the sensor image (bs,nx,ny)
                    shift_info (dict): The fourier kernel to do dispersion.
                    normalize (bool): it makes sense to integrate the mask and normalize with it. However in practice it doesnt work well.

            Returns:
                    (array): undispersed, unintegrated cube
    '''
     
    nc = psi.shape[1]
    kernel = shift_info['kernel']

    if normalize:
        divider = (torch.sum(psi,axis=1)) #will break if psi has rows of 0s
        divider[divider==0] = 1
        g = g / divider

    cube = psi * torch.tile(g.unsqueeze(1),(1,nc,1,1))

    cube_disp = disperser.undisperse_all_orders(cube,kernel,wiener=wiener,lamb=lamb)

    return cube_disp


class disperser:
    
    @staticmethod
    def disperse_all_orders(cube, kernel, pad=False): # @TODO Why does padding break?

        # the kernel comes in its real space form.
        kernel = torch.fft.fftshift(kernel,dim=(2,3)) #dont get why this is neccessary really

        nx,ny = cube.shape[2:]
        if pad:
            pad_x = int(np.ceil(nx/2))
            pad_y = int(np.ceil(ny/2))
            cube = torch.nn.functional.pad(cube,(pad_y,pad_y,pad_x,pad_x))
            kernel = torch.nn.functional.pad(kernel,(pad_y,pad_y,pad_x,pad_x))

        f_cube = torch.fft.fft2(cube)
        f_kernel = torch.fft.fft2(kernel)
        f_cube_disp = f_cube * f_kernel
        cube_disp = torch.real(torch.fft.ifft2(f_cube_disp)) #or abs?

        if pad:
            cube_disp = cube_disp[:,:,pad_x:-pad_x,pad_y:-pad_y]
        return cube_disp
    

    @staticmethod
    def undisperse_all_orders(cube, kernel, wiener = False, pad=False, **kwargs): # @TODO Why does padding break? 

        kernel = torch.fft.fftshift(kernel,dim=(2,3)) #dont get why this is neccessary really

        nx,ny = cube.shape[2:]
        if pad:
            pad_x = int(np.ceil(nx/2))
            pad_y = int(np.ceil(ny/2))
            cube = torch.nn.functional.pad(cube,(pad_y,pad_y,pad_x,pad_x))
            kernel = torch.nn.functional.pad(kernel,(pad_y,pad_y,pad_x,pad_x))

        f_cube = torch.fft.fft2(cube)
        f_kernel = torch.fft.fft2(kernel)

        if wiener:
            lamb = kwargs['lamb']
            f_cube_undisp=(torch.conj(f_kernel)*f_cube)/ (torch.square(torch.abs(f_kernel))+ lamb)#, dtype=tf.complex64)
        else:
            f_cube_undisp = f_cube / f_kernel

        cube_undisp=torch.real((torch.fft.ifft2(f_cube_undisp))) #or abs


        if pad:
            cube_undisp = cube_undisp[:,:,pad_x:-pad_x,pad_y:-pad_y]

        return cube_undisp
        

