# -*- coding: utf-8 -*-

"""
@author: Yunfei Zhang
@software: PyCharm
@file: Models.py
"""
import numpy as np
import torch
import torch.nn as nn



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Discriminator(nn.Module):
    '''Discriminator with PatchGAN'''

    def __init__(self, conv_dim, layer_num):
        super(Discriminator, self).__init__()

        layers = []

        # input layer
        layers.append(nn.Conv2d(1+1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_dim = conv_dim

        # hidden layers
        for i in range(1, layer_num):
            layers.append(nn.Conv2d(current_dim, current_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.InstanceNorm2d(current_dim*2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_dim *= 2

        self.model = nn.Sequential(*layers)

        # output layer
        self.conv_src = nn.Sequential(nn.Conv2d(current_dim, 1, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.Tanh())

    def forward(self, x, c):
        x = self.model(torch.cat([x, c], dim=1))
        out_src = self.conv_src(x)
        return out_src











class ResNetBlock(nn.Module):
    def __init__(self, in_size, out_size, normalize=[True,True,True], dropout=0.0):
        self.in_size = in_size
        self.out_size = out_size
        super(ResNetBlock, self).__init__()

        layers = []
        layers.append(nn.LeakyReLU(0.2))
        if normalize[0]:
            layers.append(nn.InstanceNorm2d(out_size))

        layers.append(nn.Conv2d(in_size, out_size, kernel_size=1,  bias=False))




        layers.append(nn.LeakyReLU(0.2))
        if normalize[1]:
            layers.append(nn.InstanceNorm2d(out_size))


        layers.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=False))




        layers.append(nn.LeakyReLU(0.2))
        if normalize[2]:
            layers.append(nn.InstanceNorm2d(out_size))


        layers.append(nn.Conv2d(out_size, out_size, kernel_size=1, bias=False))



        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

        self.identity_layer = nn.Conv2d(in_size,out_size,kernel_size=1,bias=False)

    def forward(self, x):
        if self.in_size == self.out_size:
            return self.model(x)+x
        else:
            return self.model(x)+self.identity_layer(x)






class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        self.normalize = list(np.ones(3)*normalize)

        self.ResNetBlock = ResNetBlock(in_size,out_size,normalize=self.normalize,dropout=dropout)

        self.ConvDown = nn.Conv2d(out_size, out_size, kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.ConvDown(self.ResNetBlock(x))




class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetUp, self).__init__()
        self.normalize = list(np.ones(3) * normalize)

        self.ResNetBlock = ResNetBlock(in_size,out_size,normalize=self.normalize,dropout=dropout)
        self.ConvUp = nn.ConvTranspose2d(out_size, out_size, 4, 2, 1, bias=False)



    def forward(self, x, skip_input):
        x = self.ConvUp(self.ResNetBlock(x))
        x = torch.cat((x, skip_input), 1)

        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.3)
        self.down5 = UNetDown(512, 512, dropout=0.3)
        self.down6 = UNetDown(512, 512, dropout=0.3)
        self.down7 = UNetDown(512, 512, dropout=0.3)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.3)

        self.up1 = UNetUp(512, 512, normalize=False, dropout=0.3)
        self.up2 = UNetUp(1024, 512, dropout=0.3)
        self.up3 = UNetUp(1024, 512, dropout=0.3)
        self.up4 = UNetUp(1024, 512, dropout=0.3)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)




        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)









