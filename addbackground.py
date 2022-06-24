import cv2
import glob
from PIL import Image
import numpy as np
import os 
from pathlib import Path
import imp
import copy
def ib(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255
    print(v.shape)
    tam=np.int16(v)
    tam+=np.int16(value/2)
    
    tam[tam > lim] = 255
    tam[tam < 0] = 0
    v=np.uint8(tam)
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img
def exba(imgeff,imgmain):
    eff = np.average(imgeff)
    mai = np.average(imgmain)*0.5
    imgout = ib(imgeff,value=mai-eff)
    return imgout
def square(img):
    s = min(img.shape[0:2])
    y,x,z = img.shape
    startx = x//2-(s//2)
    starty = y//2-(s//2)    
    return img[starty:starty+s,startx:startx+s]
background = [img for img in glob.glob("background/*.jpg")]
image = [img for img in glob.glob("predt/*.jpg")]
mask = copy.copy(image)
for i in range(len(mask)):
    mask[i]=mask[i].split('\\',1)
    mask[i]=mask[i][1]
    mask[i]='mask\\'+mask[i][0:(len(mask[i])-5)]+'1.jpg'

siz = len(image)
print(siz)
for i in range(siz):
    img= cv2.imread(image[i])
    msk = cv2.imread(mask[i])
    bck = cv2.imread(background[i])
    bck = square(bck)
    bck = cv2.resize(bck, (128,128), interpolation = cv2.INTER_AREA)
    bck = exba(bck,img)
    st = image[i].split('\\',1)
    st = st[1]
    for j in range(img.shape[0]):
        for k in range(img.shape[1]):
            if (msk[j,k,0]<100):
                img[j,k]=bck[j,k]
    cv2.imwrite('dttrain\\'+st,img)
    print(st)
