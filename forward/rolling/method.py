'''
This file contains the physics-based forward model and its transpose.
'''

import numpy as np
import torch


def calc_psi_z(psi,z,shift_info):
    '''
    Simulate the forward sensing process

            Parameters:
                    psi (array): The mask
                    z (array): the cube
                    shift_info (dict): dictionary containing the dispersion information

            Returns:
                    (array): dispersed, integrated cube
    '''

    dispersions = shift_info['dispersions']

    z_disp = disperser.disperse_all_orders(z,dispersions)

    return torch.sum(psi*z_disp,axis=1)


def calc_psiT_g(psi,g,shift_info):
    '''
    Transpose of the sensing process.

            Parameters:
                    psi (array): The mask
                    g (array): the sensor image
                    shift_info (dict): dictionary containing the dispersion information

            Returns:
                    (array): undispersed, unintegrated cube
    '''
     
    nc = psi.shape[1]

    g = g / (torch.sum(psi,axis=1)) #will break if psi has rows of 0s

    cube = psi * torch.tile(g[:,:,:,torch.newaxis],(1,nc,1,1))

    dispersions = shift_info['dispersions']

    cube_disp = disperser.undisperse_all_orders(cube,dispersions)

    return cube_disp



class disperser:
    '''
    This class contains the functions to disperse all of the orders of the cube (bs,nc,nx,ny)
    '''
    @staticmethod
    def disperse_all_orders(cubes, dispersions, split = [755,1405], orders=[-1,0,1]):
        
        all_levels = [0,split[0],split[1],cubes.shape[-1]]

        all_orders = []

        for n,i in enumerate(orders):
            cube_slice = cubes[...,all_levels[n]:all_levels[n+1]]
            all_orders.append(disperser.disperse_order(cube_slice,dispersions,order=i))
        
        
        dispersed_cube = torch.cat(all_orders,axis=-1)

        return dispersed_cube
    
    @staticmethod
    def undisperse_all_orders(cubes, dispersions, split = [755,1405], orders=[-1,0,1]):
        
        undispersed_cube = disperser.disperse_all_orders(cubes, -dispersions, split = split, orders=orders)
        return undispersed_cube


    @staticmethod
    def disperse_order(cube_slice, dispersions, order = -1):
        '''
        cube_slice is the area of the sensor corresponding to the order.
        d is a list of dispersions. 
        '''

        if order not in [-1,0,1]:
            raise ValueError('Order must be -1,0,1.')

        collect_m = []

        for ch,d in enumerate(dispersions):
            
            if d > 0:
                if order==-1:
                    m_mean = disperser.disperse_left(cube_slice[:,ch],d)

                elif order==1:
                    m_mean = disperser.disperse_right(cube_slice[:,ch],d)
                
            elif d<0:
                d = -d
                if order==-1:
                    m_mean = disperser.disperse_right(cube_slice[:,ch],d)
                elif order==1:
                    m_mean = disperser.disperse_left(cube_slice[:,ch],d)
            
            if order == 0:
                m_mean = cube_slice[:,ch]
            
            collect_m.append(m_mean)

            
        return torch.stack(collect_m,axis=1)


    @staticmethod
    def disperse_left(cube,dispersion):
        bs,nx,ny = cube.shape
        d_up =  int(dispersion + 1) 
        d_down = int(dispersion)


        m1 = torch.cat(  (cube[:,:,d_up:], torch.zeros((bs,nx,d_up)) ) ,axis=2)  
        if d_down!=0:
            m2 = torch.cat(  (cube[:,:,d_down:], torch.zeros((bs,nx,d_down)) ) ,axis=2)
        else:
            m2 = cube

        m_mean = m1 * (1 - (d_up-dispersion))/(d_up-d_down) + m2 * (1 - (dispersion - d_down))/(d_up-d_down)
        
        return m_mean
    
    
    @staticmethod
    def disperse_right(cube,dispersion):
        bs,nx,ny = cube.shape
        d_up =  int(dispersion + 1) 
        d_down = int(dispersion)

        m2 =torch.cat(  (torch.zeros((bs, nx,d_up)),  cube[:,:,:-d_up] ),axis=2)
        if d_down!=0:
            m1 =torch.cat(  (torch.zeros((bs, nx,d_down)),  cube[:,:,:-d_down] ),axis=2)
        else:
            m1 = cube

        m_mean = m1 * (d_up-dispersion)/(d_up-d_down) + m2 * (dispersion - d_down)/(d_up-d_down)

        return m_mean




