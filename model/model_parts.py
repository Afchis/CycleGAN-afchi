import torch
import torch.nn as nn


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvRelu, self).__init__()
        self.convrelu = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                                      nn.BatchNorm2d(out_channels),
                                      nn.ReLU())

    def forward(self, x):
        return self.convrelu(x)


class UpAndCat(nn.Module):
    def __init__(self):
        super(UpAndCat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x_up, x_cat):
        out = self.up(x_up)
        out = torch.cat([out, x_cat], dim=1)
        return out


class Cat(nn.Module):    
    def __init__(self):
        super(Cat, self).__init__() 

    def forward(self, x_up, x_cat):
        out = torch.cat([x_up, x_cat], dim=1)
        return out