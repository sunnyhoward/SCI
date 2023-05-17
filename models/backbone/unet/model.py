""" Full assembly of the parts to form the complete network """

from .modules import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_levels, bilinear=False, BN=False, ):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        clist = [64,128,256,512,1024]

        self.inc = (DoubleConv(n_channels, clist[0],BN=BN)) #input convolution


        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        factor = 2 if bilinear else 1

        for i in range(n_levels-1): 
            if i!=n_levels-2:
                self.downs.append(Down(clist[i], clist[i+1],BN=BN))
            else:
                self.downs.append(Down(clist[i], clist[i+1]//factor,BN=BN))

            self.ups.append(Up(clist[n_levels-1-i], clist[n_levels-1 -i-1]//factor, bilinear,BN=BN))

        self.outc = (OutConv(clist[0], n_classes))




    def forward(self, x):
        
        x = self.inc(x)

        x_skips = []
        for i in range(len(self.downs)):
            x_skips.append(x)
            x = self.downs[i](x)

        for i in range(len(self.ups)):
            x = self.ups[i](x, x_skips[-i-1])
    
        y = self.outc(x)
        return y

    # def use_checkpointing(self):
    #     self.inc = torch.utils.checkpoint(self.inc)
    #     self.down1 = torch.utils.checkpoint(self.down1)
    #     self.down2 = torch.utils.checkpoint(self.down2)
    #     self.down3 = torch.utils.checkpoint(self.down3)
    #     self.down4 = torch.utils.checkpoint(self.down4)
    #     self.up1 = torch.utils.checkpoint(self.up1)
    #     self.up2 = torch.utils.checkpoint(self.up2)
    #     self.up3 = torch.utils.checkpoint(self.up3)
    #     self.up4 = torch.utils.checkpoint(self.up4)
    #     self.outc = torch.utils.checkpoint(self.outc)