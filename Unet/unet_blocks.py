import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as G


class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.double_conv=nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
            
    def forward(self,x):
        return self.double_conv(x)
        
        
class DownSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.down_sample=nn.Sequential(
            nn.MaxPool2d(2,stride=2),
            DoubleConv(in_channels,out_channels)
        )
        
    def forward(self,x):
        return self.down_sample(x)

    
class UpSample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.up=nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)
        self.double_conv=DoubleConv(in_channels,out_channels)
        
    def forward(self,x,y):
        x=self.up(x)
        x_size=x.size()[2]
        y_size=y.size()[2]
        s_orign=(y_size-x_size)//2
        y=y[:,:,s_orign:s_orign+x_size,s_orign:s_orign+x_size]
        x=torch.cat([x,y],1)
        return self.double_conv(x)
        
        
class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1)
        
    def forward(self,x):
        return self.conv(x)


