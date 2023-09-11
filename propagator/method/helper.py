    
import torch
import numpy as np


def propagate(cube, kernel, pad): 
    wl = cube.shape[1]
    return torch.concat([_propagate(cube[:,c:c+1],kernel[:,c:c+1], pad) for c in range(wl)],axis=1) #to avoid making big tensors for memory problems

def _propagate(cube, kernel, pad):
    '''
    pad is fraction of the image size on each side.
    '''
    nx,ny = cube.shape[2:]

    pad_x = int(pad * nx)
    pad_y = int(pad * ny)

    cube = torch.nn.functional.pad(cube,(pad_y,pad_y,pad_x,pad_x))
    kernel = torch.nn.functional.pad(kernel,(pad_y,pad_y,pad_x,pad_x))

    # the kernel comes in its real space form.
    kernel = torch.fft.fftshift(kernel,dim=(2,3)) #dont get why this is neccessary really


    f_cube = torch.fft.fft2(cube)
    f_kernel = torch.fft.fft2(kernel)
    f_cube_disp = f_cube * f_kernel
    cube_disp = torch.fft.ifft2(f_cube_disp) #or abs?

    if (pad_x > 0) and (pad_y > 0):
        cube_disp = cube_disp[:,:,pad_x:-pad_x,pad_y:-pad_y]
    
    return cube_disp