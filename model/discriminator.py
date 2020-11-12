import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_parts import ConvRelu

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)


class Discriminator1(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
                                           nn.ReLU(),
                                           ConvRelu(64, 128, kernel_size=4, stride=2, padding=1),
                                           ConvRelu(128, 256, kernel_size=4, stride=2, padding=1),
                                           ConvRelu(256, 512, kernel_size=4, stride=1, padding=1),
                                           nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
                                           nn.Sigmoid())

    def forward(self, x):
        out = self.discriminator(x)
        return F.avg_pool2d(out, out.size()[2:]).view(out.size()[0], -1)


if __name__ == "__main__":
    x = torch.rand([16, 3, 256, 256])
    model = Discriminator()
    out = model(x)
    print(out.shape)