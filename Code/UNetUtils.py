#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self,in_channels,out_channels,mid_channels=None) -> None:
        super(DoubleConv,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=False),
            nn.ReLU(inplace=True),
            #nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False),
            #nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.double_conv(x)
    
class Down(nn.Module):
    
    def __init__(self,in_channels,out_channels) -> None:
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )
    def forward(self,x):
        return self.maxpool_conv(x)
    
class UpTransPose(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=2,stride=2) -> None:
        super(UpTransPose,self).__init__()
        self.up = nn.ConvTranspose2d(in_channels,in_channels//2,
                                     kernel_size=kernel_size,stride=stride)
        self.conv = DoubleConv(in_channels,out_channels)

    def forward(self,x1,x2):
        x1 = self.up(x1)
        diffY = x1.size()[2] - x2.size()[2]
        diffX = x1.size()[3] - x2.size()[3]

        x2 = F.pad(x2,[diffX//2, diffX-diffX//2,
                       diffY//2, diffY-diffY//2])
        x = torch.cat([x1,x2],dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self,in_channels,out_channels) -> None:
        super(OutConv,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        return self.relu(self.conv(x))



