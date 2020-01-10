import torch.nn as nn
from .unet_blocks import *

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.initial_block=DoubleConv(1,64)
        self.down1=DownSample(64,128)
        self.down2=DownSample(128,256)
        self.down3=DownSample(256,512)
        self.down4=DownSample(512,1024)
        self.up1=UpSample(1024,512)
        self.up2=UpSample(512,256)
        self.up3=UpSample(256,128)
        self.up4=UpSample(128,64)
        self.final_conv=OutConv(64,2)
        
    def forward(self,x):
        x1=self.initial_block(x)
        x2=self.down1(x1)
        x3=self.down2(x2)
        x4=self.down3(x3)
        x=self.down4(x4)
        x=self.up1(x,x4)
        x=self.up2(x,x3)
        x=self.up3(x,x2)
        x=self.up4(x,x1)
        x=self.final_conv(x)
        x=F.softmax(x,dim=1)
        
        return x
