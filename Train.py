# -*- coding: utf-8 -*-

"""
@author: Yunfei Zhang
@software: PyCharm
@file: Train.py
"""



import os

import PIL.Image
import matplotlib.pyplot as plt


from Models import Generator, Discriminator

from Loss_Functions import *

import torch
from torch import nn
from torch.utils import data
import numpy as np
from torch import optim
import glob
import itertools
import torchvision
import PIL







# DataLoader:
Batch_Size = int(input("Please input the batch size:"))

########
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


class my_ds(data.Dataset):
    def __init__(self, x_path, y_path, transforms):
        super(my_ds, self).__init__()
        self.x_path = x_path
        self.y_path = y_path
        self.transforms = transforms

    def __getitem__(self, item):
        x = self.x_path[item]
        y = self.y_path[item]
        x_img = PIL.Image.open(x)
        y_img = PIL.Image.open(y)
        x_tensor = self.transforms(x_img)
        x_tensor = x_tensor * 2 - 1
        y_tensor = self.transforms(y_img)
        y_tensor = y_tensor * 2 - 1

        return x_tensor, y_tensor

    def __len__(self):
        return len(self.x_path)


def make_dl(ds, batch_size=10, shuffle=True):
    return data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)




x_path_ls = glob.glob("Pre\*.jpg") #x_path_ls is a list containing all file path of pre-contrast images
y_path_ls = glob.glob("Target\*.jpg") #y_path_ls is a list containing all file path of images at specific enhanced phase




ds = my_ds(x_path_ls, y_path_ls, transforms=transforms)
dl = make_dl(ds, batch_size=1)  ############训练时建议设置为10,绘制模型时建议设置为1




# Model Train:

Num_Eppches = int(input("Please input the number of epoches:"))
Decay_Epoch = int(input("Please input the number of decay epoches:"))
lr_D = 0.0001
lr_G = 0.00005

date = "20230201"

G_B = Generator(1, 1).cuda()  # Output is the prediction of target, x==>>y

D_B = Discriminator(8, 5).cuda()


loss_D_B_epoch = []
loss_G_epoch = []

for epoch in range(1, Num_Eppches + 1):

    d_b_loss = []
    g_loss = []
    if epoch > Decay_Epoch:

        lr_G = lr_G / 1.2

        lr_D = lr_D / 1.2

        if lr_G >= 1e-8:
            lr_G = lr_G
        else:
            lr_G = 1e-8

        if lr_D >= 1e-8:
            lr_D = lr_D
        else:
            lr_D = 1e-8

        G_optimizer = torch.optim.Adam(G_B.parameters(), lr_G)

        D_B_optimizer = optim.Adam(D_B.parameters(), lr_D - lr_D / (2 + epoch - Decay_Epoch))

    else:
        G_optimizer = torch.optim.Adam(G_B.parameters(),
                                       lr=lr_G)
        D_B_optimizer = optim.Adam(D_B.parameters(), lr_D)

    index_dl = 0
    for x_real, y_real in dl:
        index_dl += 1
        x_real = x_real.cuda()
        y_real = y_real.cuda()

        # train Discriminator B:
        y_fake = G_B(x_real)
        y_fake_logit = D_B(y_fake, y_fake)
        y_real_logit = D_B(y_real, y_real)
        D_B_Loss = (Loss_real_BCE(y_real_logit) + Loss_fake_BCE(y_fake_logit)) / 2

        D_B_optimizer.zero_grad()
        D_B_Loss.backward()
        D_B_optimizer.step()

        # Record Loss:
        d_b_loss_temp = D_B_Loss.detach().cpu().numpy()
        d_b_loss.append(d_b_loss_temp)

        # train Generator:

        for i in range(2):

            # GAN Loss

            y_fake = G_B(x_real)

            y_fake_logit = D_B(y_fake, y_fake)

            GAN_Loss = Loss_real_BCE(y_fake_logit)
            identity_loss = L2_Loss(y_real, y_fake)

            Total_G_Loss = GAN_Loss + 10 * identity_loss

            G_optimizer.zero_grad()
            Total_G_Loss.backward()
            G_optimizer.step()

        g_loss_temp = Total_G_Loss.detach().cpu().numpy()
        g_loss.append(g_loss_temp)

        if index_dl % 20 == 0:
            d_b_loss_dl = np.mean(d_b_loss[index_dl - 20:])
            g_loss_dl = np.mean(g_loss[index_dl - 20:])
            print("This is %d Epoch,%d iterations，the loss of discriminator is %.4f,the loss of generator is %.4f" % (epoch, index_dl, d_b_loss_dl, g_loss_dl))

    loss_D_B_epoch.append(np.mean(d_b_loss))
    loss_G_epoch.append(np.mean(g_loss))

    if (epoch + 1) % 5 == 0:
        G_Name = date + "_G_D_epoch" + str(epoch + 1) + ".pth"
        D_Name = date + "_D_D_epoch" + str(epoch + 1) + ".pth"
        torch.save(obj=G_B.state_dict(), f=G_Name)
        torch.save(obj=D_B.state_dict(), f=D_Name)
















