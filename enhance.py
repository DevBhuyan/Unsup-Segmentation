#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 02:02:34 2023

@author: devn
"""
import cv2
import os
from tqdm import tqdm

def clahe(image, clipLimit):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(16, 16))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    image = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    return image

def dullrazor(clahe_image):
    img = clahe_image
    grayScale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
    kernel = cv2.getStructuringElement(1,(9,9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    bhg= cv2.GaussianBlur(blackhat,(3,3),cv2.BORDER_DEFAULT)
    ret,mask = cv2.threshold(bhg,10,255,cv2.THRESH_BINARY)
    dst = cv2.inpaint(img,mask,6,cv2.INPAINT_TELEA)
    return dst

def enhance(img, clipLimit = 2.0):
    return dullrazor(clahe(img, clipLimit))

if __name__ =="__main__":
    src = './ISIC 2016 original resized/'
    dest = './ISIC 2016 original resized CLAHE ClipLimit 2/'
    if not os.path.exists(dest):
        os.makedirs(dest)
    for folder in os.listdir(src):
        print('folder : ', src+folder+'/')
        for subfolder in os.listdir(src+folder+'/'):
            print('type : ', src+folder+'/'+subfolder+'/')
            for file in tqdm(os.listdir(src+folder+'/'+subfolder+'/'), total=len(os.listdir(src+folder+'/'+subfolder+'/'))):
                out = enhance(cv2.imread(src+folder+'/'+subfolder+'/'+file, cv2.IMREAD_COLOR))
                if not os.path.exists(dest+folder+'/'+subfolder+'/'):
                    os.makedirs(dest+folder+'/'+subfolder+'/')
                cv2.imwrite(dest+folder+'/'+subfolder+'/'+file, out)