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




def findclusters(kernel, padding = 4, verbose = False):
    '''
    This function identifies the regions of the kernel that will be made trainable
    '''
    sx,sy = kernel.shape[2:]

    a = torch.sum(kernel[0].detach().cpu(),dim=0)
    x,y = np.where(a>0)

    xclusters = []
    xstart = x[0]
    for i in range(1,len(x)):
        if np.abs(x[i] - x[i-1]) > 10:
            xclusters.append((xstart,x[i-1]))
            xstart = x[i]
    xclusters.append((xstart,x[-1]))


    # now for each x cluster, how many y clusters do we have?
    clusters = np.zeros((2,2,9),dtype=int)
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
                
                clusters[0,0,n] = xcluster[0] - padding if xcluster[0] - padding >= 0 else 0
                clusters[0,1,n] = xcluster[1] + padding if xcluster[1] + padding <= sx-1 else sx-1
                clusters[1,0,n] = ycluster[0] - padding if ycluster[0] - padding >= 0 else 0
                clusters[1,1,n] = ycluster[1] + padding if ycluster[1] + padding <= sy-1 else sy-1
                n += 1

                ystart = yvalues[i+1] #the beginning of the next cluster

        ycluster = (ystart,yvalues[i+1])
        if verbose:
            print(xcluster, ycluster)

        clusters[0,0,n] = xcluster[0] - padding if xcluster[0] - padding >= 0 else 0
        clusters[0,1,n] = xcluster[1] + padding if xcluster[1] + padding <= sx-1 else sx-1
        clusters[1,0,n] = ycluster[0] - padding if ycluster[0] - padding >= 0 else 0
        clusters[1,1,n] = ycluster[1] + padding if ycluster[1] + padding <= sy-1 else sy-1
        n += 1
        # print(n)

    return  clusters