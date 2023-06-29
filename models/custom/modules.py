import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CoordGate(nn.Module):
    def __init__(self, encoding_layers, enc_channels, out_channels, size:list=[256,256],device='cuda'):
        super(CoordGate, self).__init__()

        x_coord, y_coord = torch.linspace(-1,1,int(size[0])), torch.linspace(-1,1,int(size[1]))

        # self.pos = torch.stack(torch.meshgrid((x_coord,y_coord), indexing='ij'), dim=-1).view(-1,2).to(device)
        self.register_buffer('pos', torch.stack(torch.meshgrid((x_coord,y_coord), indexing='ij'), dim=-1).view(-1,2))#.to(device)
        

        self.encoder = nn.Sequential()
        for i in range(encoding_layers):
            if i == 0:
                self.encoder.add_module('linear'+str(i),nn.Linear(2,enc_channels))
            elif i == encoding_layers-1:
                self.encoder.add_module('linear'+str(i),nn.Linear(enc_channels,out_channels))
            else:
                self.encoder.add_module('linear'+str(i),nn.Linear(enc_channels,enc_channels))
        
        self.conv = nn.Conv2d(out_channels,out_channels,1,padding='same')


    def forward(self, x):
        '''
        x is (bs,nc,nx,ny)
        '''

        encoded_pos = self.encoder(self.pos).view(1,x.shape[2],x.shape[3],x.shape[1]).permute(0,3,1,2)

        x = x * encoded_pos

        return self.conv(x)




def findclusters(kernel, padding = 4, threshold=0, type = 'nearest_neighbour', verbose = False):
    '''
    This function identifies the regions of the kernel.

    threshold is fraction of maximum value.

    type: nearest neighbour is for creating a mask, with padding around all the places where the kernel is nonzero. (useful for trainable kernel)
    type: boxes is for finding the coordinates of boxes around each part of the kernel. output is shape (9, 2, 2): [no_regions, (x/y), (start/end)]
    '''
    
    


    if type == 'nearest_neighbour':
        sc,sx,sy = kernel[0].shape
        rel_threshold = kernel.max() * threshold
        c,x,y = torch.where(kernel[0]>rel_threshold)
        mask = torch.zeros((sc,sx,sy),dtype=bool) #this will be used to index a kernel.

        for i in range(len(x)):

            lb_x = x[i] - padding if x[i] - padding >= 0 else 0
            ub_x = x[i] + padding if x[i] + padding <= sx-1 else sx-1
            lb_y = y[i] - padding if y[i] - padding >= 0 else 0
            ub_y = y[i] + padding if y[i] + padding <= sy-1 else sy-1
            mask[c[i],lb_x:ub_x,lb_y:ub_y] = True

        return  mask

    elif type == 'boxes':

        sx,sy = kernel.shape[2:]
        a = torch.sum(kernel[0],dim=0)
        rel_threshold = a.max() * threshold
        x,y = np.where(a>rel_threshold)

        xclusters = []
        xstart = x[0]
        for i in range(1,len(x)):
            if np.abs(x[i] - x[i-1]) > 10:
                xclusters.append((xstart,x[i-1]))
                xstart = x[i]
        xclusters.append((xstart,x[-1]))

        # now for each x cluster, how many y clusters do we have?
        clusters = torch.zeros((9,2,2),dtype=int)
        n = 0

        for xcluster in xclusters:
            cond = (x>=xcluster[0]) & (x<=xcluster[1])
            yvalues = y[cond]
            yvalues = np.sort(yvalues)
            ystart = yvalues[0] #the beginning of the cluster
            for i in range(len(yvalues)-1):
                if np.abs(yvalues[i+1] - yvalues[i]) > 10: #if the next point is more than 10 away.
                    ycluster = (ystart,yvalues[i]) #save the cluster
                    if verbose:
                        print(xcluster, ycluster)
                    
                    clusters[n,0,0] = xcluster[0] - padding if xcluster[0] - padding >= 0 else 0
                    clusters[n,0,1] = xcluster[1] + padding if xcluster[1] + padding <= sx-1 else sx-1
                    clusters[n,1,0] = ycluster[0] - padding if ycluster[0] - padding >= 0 else 0
                    clusters[n,1,1] = ycluster[1] + padding if ycluster[1] + padding <= sy-1 else sy-1
                    n += 1

                    ystart = yvalues[i+1] #the beginning of the next cluster

            ycluster = (ystart,yvalues[i+1])
            if verbose:
                print(xcluster, ycluster)

            clusters[n,0,0] = xcluster[0] - padding if xcluster[0] - padding >= 0 else 0
            clusters[n,0,1] = xcluster[1] + padding if xcluster[1] + padding <= sx-1 else sx-1
            clusters[n,1,0] = ycluster[0] - padding if ycluster[0] - padding >= 0 else 0
            clusters[n,1,1] = ycluster[1] + padding if ycluster[1] + padding <= sy-1 else sy-1
            n += 1
            # print(n)

    

        return clusters










################## Losses & Metrics #####################
        
        

class TVLoss(nn.Module):
    def __init__(self, mse_weight=1, tv_weight=0.01):
        super(TVLoss, self).__init__()
        self.mse_weight = mse_weight
        self.tv_weight = tv_weight

    def forward(self, prediction, target, model):
        mse_loss = nn.MSELoss()(prediction, target) 

        var_ker = model.fill_kernel()

        # Calculate the Total Variation (TV) loss
        tv_loss = self.total_variation_loss(var_ker)

        # Combine the MSE and TV losses
        loss = self.mse_weight * mse_loss + self.tv_weight * tv_loss

        return loss

    def total_variation_loss(self, x):
        # Calculate the Total Variation (TV) loss
        norm = torch.prod(torch.tensor(x.shape))
        
        # Calculate the horizontal and vertical gradients
        tv_loss = ( torch.sum((x[:, :-1, :, :] - x[:, 1:, :, :])**2) + torch.sum((x[:, :,  :-1, :] - x[:, :, 1:, :])**2)  +  torch.sum((x[:, :, :, :-1] - x[:, :, :, 1:])**2) ) / norm

        return tv_loss
    



class CenterOfMassLoss(nn.Module):
    '''
    A model that tries to match the center of mass of the predicted and target tensors in certain regions.
    '''
    def __init__(self, region, intensity_factor=3):
        super(CenterOfMassLoss, self).__init__()
        self.region = region
        self.intensity_factor = intensity_factor

    def forward(self, predicted, target):

        total_loss = 0

        for i in range(8): 
            for l in range(21):
                # Calculate the center of mass for the predicted and target tensors
                pred_center = self.calculate_center_of_mass(predicted[:,l], region = self.region[i])
                target_center = self.calculate_center_of_mass(target[:,l], region = self.region[i])

                # Calculate the loss as the Euclidean distance between the centers of mass
                total_loss  += torch.norm(pred_center - target_center, p=2) / 21

        return total_loss


    def calculate_center_of_mass(self, tensor, region):

        factor = self.intensity_factor

        # Calculate the center of mass within the specified region of the tensor
        region_tensor = tensor[:,  region[0,0]:region[0,1], region[1,0]:region[1,1]] ** factor

        sum_tensor_x = torch.sum(region_tensor, dim=(2))
        sum_tensor_y = torch.sum(region_tensor, dim=(1))

        indices_x = torch.arange(region_tensor.size(1), device=tensor.device)
        indices_y = torch.arange(region_tensor.size(2), device=tensor.device)

        center_x = torch.sum(indices_x.unsqueeze(0) * sum_tensor_x, dim=1) / torch.sum(sum_tensor_x,dim=1)
        center_y = torch.sum(indices_y.unsqueeze(0) * sum_tensor_y, dim=1) / torch.sum(sum_tensor_y,dim=1)



        center_of_mass = torch.stack((center_x, center_y), dim=1)

        return center_of_mass
    



