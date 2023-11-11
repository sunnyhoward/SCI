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
    def disperse_all_orders(cube, kernel, pad=True, real_part=True): 
        wl = cube.shape[1]
        return torch.concat([disperser._disperse_all_orders(cube[:,c:c+1],kernel[:,c:c+1], pad, real_part) for c in range(wl)],axis=1) #to avoid making big tensors for memory problems

    @staticmethod
    def _disperse_all_orders(cube, kernel, pad=True,real_part=True):
        nx,ny = cube.shape[2:]
        if pad:
            pad_x = int(np.ceil(nx/2)) if nx!=1 else 0
            pad_y = int(np.ceil(ny/2))

            cube = torch.nn.functional.pad(cube,(pad_y,pad_y,pad_x,pad_x))
            kernel = torch.nn.functional.pad(kernel,(pad_y,pad_y,pad_x,pad_x))

        # the kernel comes in its real space form.
        kernel = torch.fft.fftshift(kernel,dim=(2,3)) #dont get why this is neccessary really


        f_cube = torch.fft.fft2(cube)
        f_kernel = torch.fft.fft2(kernel)
        f_cube_disp = f_cube * f_kernel
        cube_disp = torch.fft.ifft2(f_cube_disp) #or abs?

        if real_part:
            cube_disp = torch.real(cube_disp)
            cube_disp = torch.clamp(cube_disp, min=0)

        if pad:
            cube_disp = cube_disp[:,:,pad_x:-pad_x,pad_y:-pad_y] if pad_x!=0 else cube_disp[:,:,:,pad_y:-pad_y]
        return cube_disp
    


    @staticmethod
    def undisperse_all_orders(cube, kernel, wiener = False, pad=True, real_part = True,  **kwargs):
        wl = cube.shape[1]
        return torch.concat([disperser._undisperse_all_orders(cube[:,c:c+1],kernel[:,c:c+1], wiener, pad, real_part, **kwargs) for c in range(wl)],axis=1)

    @staticmethod
    def _undisperse_all_orders(cube, kernel, wiener = False, pad=True, real_part = True,  **kwargs):
        
        nx,ny = cube.shape[2:]
        if pad!=False:
            if pad==True: padfac = 0.5
            else: padfac = pad
            pad_x = int(np.ceil(padfac*nx)) if nx!=1 else 0
            pad_y = int(np.ceil(padfac*ny))

            cube = torch.nn.functional.pad(cube,(pad_y,pad_y,pad_x,pad_x))
            kernel = torch.nn.functional.pad(kernel,(pad_y,pad_y,pad_x,pad_x))

        kernel = torch.fft.fftshift(kernel,dim=(2,3)) #dont get why this is neccessary really

        f_cube = torch.fft.fft2(cube)
        f_kernel = torch.fft.fft2(kernel)

        if wiener:
            lamb = kwargs['lamb']
            f_cube_undisp=(torch.conj(f_kernel)*f_cube)/ (torch.square(torch.abs(f_kernel))+ lamb)#, dtype=tf.complex64)
        else:
            f_cube_undisp = f_cube / f_kernel

        cube_undisp=(torch.fft.ifft2(f_cube_undisp)) #or abs

        if real_part: #if not maybe we are propagating
            cube_undisp = torch.real(cube_undisp)
            cube_undisp = torch.clamp(cube_undisp, min=0)

        if pad!=False:
            cube_undisp = cube_undisp[:,:,pad_x:-pad_x,pad_y:-pad_y] if pad_x!=0 else cube_undisp[:,:,:,pad_y:-pad_y]

        return cube_undisp
        

