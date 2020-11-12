import torch
import torch.nn as nn

from .model_parts import ConvRelu, UpAndCat, Cat


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.ch_list = [3, 32, 64, 128, 256]

        ##### Down layers #####
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down1 = nn.Sequential(
            ConvRelu(self.ch_list[0], self.ch_list[1], kernel_size=3, stride=1, padding=1),
            ConvRelu(self.ch_list[1], self.ch_list[1], kernel_size=3, stride=1, padding=1)
            )
        self.down2 = nn.Sequential(
            ConvRelu(self.ch_list[1], self.ch_list[2], kernel_size=3, stride=1, padding=1),
            ConvRelu(self.ch_list[2], self.ch_list[2], kernel_size=3, stride=1, padding=1)
            )
        self.down3 = nn.Sequential(
            ConvRelu(self.ch_list[2], self.ch_list[3], kernel_size=3, stride=1, padding=1),
            ConvRelu(self.ch_list[3], self.ch_list[3], kernel_size=3, stride=1, padding=1)
            )
        self.bottom = nn.Sequential(
            ConvRelu(self.ch_list[3], self.ch_list[4], kernel_size=3, stride=1, padding=1),
            ConvRelu(self.ch_list[4], self.ch_list[4], kernel_size=3, stride=1, padding=1)
            )

        ##### Up layers #####
        self.cat_3 = Cat()
        self.up_conv_3 = nn.Sequential(
            ConvRelu(self.ch_list[4]+self.ch_list[3], self.ch_list[3], kernel_size=3, stride=1, padding=1),
            ConvRelu(self.ch_list[3],                 self.ch_list[3], kernel_size=3, stride=1, padding=1)
            )
        self.up_cat_2 = UpAndCat()
        self.up_conv_2 = nn.Sequential(
            ConvRelu(self.ch_list[3]+self.ch_list[2], self.ch_list[2], kernel_size=3, stride=1, padding=1),
            ConvRelu(self.ch_list[2],                 self.ch_list[2], kernel_size=3, stride=1, padding=1)
            )
        self.up_cat_1 = UpAndCat()
        self.up_conv_1 = nn.Sequential(
            ConvRelu(self.ch_list[2]+self.ch_list[1], self.ch_list[1], kernel_size=3, stride=1, padding=1),
            ConvRelu(self.ch_list[1],                 self.ch_list[1], kernel_size=3, stride=1, padding=1)
            )
        ##### Final layers #####
        self.final = nn.Sequential(
            nn.Conv2d(self.ch_list[1], self.ch_list[0], kernel_size=1),
            nn.Sigmoid()
            )

    def forward(self, x):
        down1_feat = self.down1(x)
        pool1 = self.pool(down1_feat)
        down2_feat = self.down2(pool1)
        pool2 = self.pool(down2_feat)
        down3_feat = self.down3(pool2)

        bottom_feat = self.bottom(down3_feat)

        up_feat3 = self.cat_3(bottom_feat, down3_feat)
        up_feat3 = self.up_conv_3(up_feat3)
        up_feat2 = self.up_cat_2(up_feat3, down2_feat)
        up_feat2 = self.up_conv_2(up_feat2)
        up_feat1 = self.up_cat_1(up_feat2, down1_feat)
        up_feat1 = self.up_conv_1(up_feat1)

        out = self.final(up_feat1)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Sigmoid() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)