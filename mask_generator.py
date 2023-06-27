#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 21:09:06 2023

@author: dev
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.init
import os
from datetime import datetime
from metrics import metrics
from tqdm import tqdm
from get_bounding_boxes import bb
from transformation_helpers import select_channels

use_cuda = torch.cuda.is_available()


base_dir = './ISIC 2016 for maskgen/'
src = base_dir+'IMAGES/'
# src = base_dir+'MASKS/UNFILTERED/'
dest = base_dir+'MASKS/'

def contrast_stretching(img):
    min_intensity = np.min(img)
    max_intensity = np.max(img)

    stretched_img = ((img - min_intensity) * (255 - 0) / (max_intensity - min_intensity))

    return stretched_img.astype(np.uint8)

def histogram_equalization_rgb(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    lab_eq = cv2.merge([l_eq, a, b])
    image_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    return image_eq

def dullrazor(img):
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(9,9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)
    return dst

def preprocess(img):
    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_R, _ = select_channels(img_RGB)
    img_R = contrast_stretching(img_R)
    img_R = (img_R) / 255.0
    img = (img-127)
    b, g, r = cv2.split(img)
    b = (b*img_R + 127).astype(np.uint8)
    g = (g*img_R + 127).astype(np.uint8)
    r = (r*img_R + 127).astype(np.uint8)

    img = (cv2.merge((b, g, r)))

    # cv2.imshow('img', img)
    # cv2.waitKey(2000)

    return img

def generator(dst):
    nChannel = 12
    maxIter = 200
    minLabels = 1
    lr = 0.001
    # lr = 0.01
    nConv = 2
    visualize = 1
    '''SET IMPORTANCE OF SIMILARITY'''
    ratio = 4
    stepsize_sim = ratio
    stepsize_con = 1/ratio

    # CNN model
    class MyNet(nn.Module):
        def __init__(self,input_dim):
            super(MyNet, self).__init__()
            self.conv1 = nn.Conv2d(input_dim, nChannel, kernel_size=3, stride=1, padding=1 )
            self.bn1 = nn.BatchNorm2d(nChannel)
            self.conv2 = nn.ModuleList()
            self.bn2 = nn.ModuleList()
            for i in range(nConv-1):
                self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
                self.bn2.append( nn.BatchNorm2d(nChannel) )
            self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
            self.bn3 = nn.BatchNorm2d(nChannel)

        def forward(self, x):
            x = self.conv1(x)
            x = F.relu( x )
            x = self.bn1(x)
            for i in range(nConv-1):
                x = self.conv2[i](x)
                x = F.relu( x )
                x = self.bn2[i](x)
            x = self.conv3(x)
            x = self.bn3(x)
            return x

    im = dst
    data = torch.from_numpy( np.array([im.transpose( (2, 0, 1) ).astype('float32')/255.]) )
    if use_cuda:
        data = data.cuda()
    data = Variable(data)

    model = MyNet( data.size(1) )
    if use_cuda:
        model.cuda()
    model.train()

    # similarity loss definition
    loss_fn = torch.nn.CrossEntropyLoss()

    # continuity loss definition
    loss_hpy = torch.nn.L1Loss(size_average = True)
    loss_hpz = torch.nn.L1Loss(size_average = True)

    HPy_target = torch.zeros(im.shape[0]-1, im.shape[1], nChannel)
    HPz_target = torch.zeros(im.shape[0], im.shape[1]-1, nChannel)
    if use_cuda:
        HPy_target = HPy_target.cuda()
        HPz_target = HPz_target.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # momentum was changed from 0.9 to 0.2
    label_colours = np.array([
                            [96, 96, 96],
                            [255, 255, 255],
                            [255, 0, 0],
                            [0, 255, 0],
                            [0, 0, 255],
                            [255, 255, 0],
                            [0, 255, 255],
                            [255, 0, 255],
                            [224, 224, 224],
                            [160, 160, 160],
                            [128, 128, 128],
                            [192, 192, 192]
        ])

    for batch_idx in range(maxIter):
        optimizer.zero_grad()
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )

        outputHP = output.reshape( (im.shape[0], im.shape[1], nChannel) )
        HPy = outputHP[1:, :, :] - outputHP[0:-1, :, :]
        HPz = outputHP[:, 1:, :] - outputHP[:, 0:-1, :]
        lhpy = loss_hpy(HPy,HPy_target)
        lhpz = loss_hpz(HPz,HPz_target)

        _, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        nLabels = len(np.unique(im_target))
        if visualize:
            im_target_rgb = np.array([label_colours[c] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            # cv2.imshow( "output", im_target_rgb )
            # cv2.waitKey(1)

        # loss
        loss = stepsize_sim * loss_fn(output, target) + stepsize_con * (lhpy + lhpz)

        loss.backward()
        optimizer.step()

        print (batch_idx, '/', maxIter, '|', ' label num :', nLabels, ' | loss :', loss.item())

        if nLabels <= minLabels:
            print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
            im_target_rgb = buffer
            break

        buffer = im_target_rgb

    # save output image
    if not visualize:
        output = model( data )[ 0 ]
        output = output.permute( 1, 2, 0 ).contiguous().view( -1, nChannel )
        _, target = torch.max( output, 1 )
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[ c % nChannel ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )

    return im_target_rgb

listdir = os.listdir(base_dir+'MASKS/UNFILTERED')

for file in tqdm(os.listdir(src), total=len(os.listdir(src))):
    # if file in listdir:
    #     continue
    start = datetime.now()
    img = cv2.imread(src+file, cv2.IMREAD_COLOR)
    dst = preprocess(dullrazor(img))
    im_target_rgb = generator(dst)
    cv2.imwrite(dest+'UNFILTERED/'+file, im_target_rgb)
    break
# bb()
# metrics()