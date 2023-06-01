import torch
import torch.nn as nn
import torch.nn.functional as F

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
