import cv2
import glob
from PIL import Image
import numpy as np
import os 
from pathlib import Path
import imp
import copy
L=256
def make_squarebw(img):
    '''
    Reshapes the non-square image by pasting
    it to the centre of a black canvas of size
    n*n where n is the biggest dimension of
    the non-square image. 
    '''
    #Getting the bigger side of the image
    s = max(img.shape[0:2])

    #Creating a white square with NUMPY  
    f = np.zeros((s,s),np.uint8)

    #Getting the centering position
    ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2

    #Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
    return f
def make_square(img):
    '''
    Reshapes the non-square image by pasting
    it to the centre of a black canvas of size
    n*n where n is the biggest dimension of
    the non-square image. 
    '''
    #Getting the bigger side of the image
    s = max(img.shape[0:2])

    #Creating a white square with NUMPY  
    f = np.zeros((s,s,3),np.uint8)

    #Getting the centering position
    ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2

    #Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
    return f
def segment(imgin):
    si = 10
    y,x,z = imgin.shape
    startx = x//2-(si//2)
    starty = y//2-(si//2)    

    imgin = cv2.fastNlMeansDenoisingColored(imgin,None,15,10,7,21)
    hsv = cv2.cvtColor(imgin, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    imgin = cv2.cvtColor(imgin, cv2.COLOR_BGR2GRAY)
    selec = s
    kp = np.average(selec[starty:(starty+si),startx:(startx+si)])
    av = np.average(selec[10:20,10:20])
    av=(kp+av)/4.4
    M, N = imgin.shape
    imgout = imgin
    avi=int(av)
    imgout[selec>avi]=255
    imgout[selec<=avi]=0
    #imgout = [255 if (h[x,y]<avi) else 0 for x,y in zip(np.arange(0,M),np.arange(0,N))]
    # for x in np.arange(0,M):
    #     for y in np.arange(0,N):
    #         if (h[x,y]<avi):
    #             imgout[x,y]=255
    #         else:
    #             imgout[x,y]=0
    return imgout
working_dir = Path()
filename=[]
for path in working_dir.glob("Dataset//Grading_dataset//Class_I//**/*.jpg"):
    filename.append(path)
count=1600
for file in filename:
    img = str(file)
    n= cv2.imread(img)
    bw=copy.copy(n)
    bw = segment(bw)
    n = make_square(n)
    n = cv2.resize(n, (128,128), interpolation = cv2.INTER_AREA)
    bw = make_squarebw(bw)
    bw = cv2.resize(bw, (128,128), interpolation = cv2.INTER_AREA)
    cv2.imwrite('predt\\'+str(count)+str(0)+'.jpg',n)
    cv2.imwrite('dttrain\\'+str(count)+str(1)+'.jpg',bw)
    cv2.imwrite('mask\\'+str(count)+str(1)+'.jpg',bw)
    print('dttrain\\'+str(count)+str(1)+'.jpg')
    count+=1