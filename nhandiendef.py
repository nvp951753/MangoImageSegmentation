import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Conv2DTranspose, Concatenate, BatchNormalization, UpSampling2D
from keras.layers import  Dropout, Activation
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K
from tensorflow.keras.utils import plot_model
import tensorflow.keras.utils as u
import glob
import random
import cv2
from random import shuffle
import random
from sklearn.model_selection import train_test_split
from pathlib import Path
import time

def mean_iou(y_true, y_pred):
    yt0 = y_true[:,:,:,0]
    yp0 = K.cast(y_pred[:,:,:,0] > 0.5, 'float32')
    inter = tf.math.count_nonzero(tf.logical_and(tf.equal(yt0, 1), tf.equal(yp0, 1)))
    union = tf.math.count_nonzero(tf.add(yt0, yp0))
    iou = tf.where(tf.equal(union, 0), 1., tf.cast(inter/union, 'float32'))
    return iou
def unet(sz = (128, 128, 3)):
  x = Input(sz)
  inputs = x
  
  #down sampling 
  f = 8
  layers = []
  
  for i in range(0, 6):
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    layers.append(x)
    x = MaxPooling2D() (x)
    f = f*2
  ff2 = 64 
  
  #bottleneck 
  j = len(layers) - 1
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
  x = Concatenate(axis=3)([x, layers[j]])
  j = j -1 
  
  #upsampling 
  for i in range(0, 5):
    ff2 = ff2//2
    f = f // 2 
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2D(f, 3, activation='relu', padding='same') (x)
    x = Conv2DTranspose(ff2, 2, strides=(2, 2), padding='same') (x)
    x = Concatenate(axis=3)([x, layers[j]])
    j = j -1 
    
  
  #classification 
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  x = Conv2D(f, 3, activation='relu', padding='same') (x)
  outputs = Conv2D(1, 1, activation='sigmoid') (x)
  
  #model creation 
  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = [mean_iou])
  
  return model
def correction(
        img,
        shadow_amount_percent, shadow_tone_percent, shadow_radius,
        highlight_amount_percent, highlight_tone_percent, highlight_radius,
        color_percent
):
    """
    Image Shadow / Highlight Correction. The same function as it in Photoshop / GIMP
    :param img: input RGB image numpy array of shape (height, width, 3)
    :param shadow_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param shadow_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param shadow_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param highlight_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param highlight_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param highlight_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param color_percent [-1.0 ~ 1.0]:
    :return:
    """
    shadow_tone = shadow_tone_percent * 255
    highlight_tone = 255 - highlight_tone_percent * 255

    shadow_gain = 1 + shadow_amount_percent * 6
    highlight_gain = 1 + highlight_amount_percent * 6

    # extract RGB channel
    height, width = img.shape[:2]
    img = img.astype(np.float)
    img_R, img_G, img_B = img[..., 2].reshape(-1), img[..., 1].reshape(-1), img[..., 0].reshape(-1)

    # The entire correction process is carried out in YUV space,
    # adjust highlights/shadows in Y space, and adjust colors in UV space
    # convert to Y channel (grey intensity) and UV channel (color)
    img_Y = .3 * img_R + .59 * img_G + .11 * img_B
    img_U = -img_R * .168736 - img_G * .331264 + img_B * .5
    img_V = img_R * .5 - img_G * .418688 - img_B * .081312

    # extract shadow / highlight
    shadow_map = 255 - img_Y * 255 / shadow_tone
    shadow_map[np.where(img_Y >= shadow_tone)] = 0
    highlight_map = 255 - (255 - img_Y) * 255 / (255 - highlight_tone)
    highlight_map[np.where(img_Y <= highlight_tone)] = 0

    # // Gaussian blur on tone map, for smoother transition
    if shadow_amount_percent * shadow_radius > 0:
        # shadow_map = cv2.GaussianBlur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius), sigmaX=0).reshape(-1)
        shadow_map = cv2.blur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius), borderType = cv2.BORDER_DEFAULT)
        shadow_map = shadow_map.reshape(-1)

    if highlight_amount_percent * highlight_radius > 0:
        # highlight_map = cv2.GaussianBlur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius), sigmaX=0).reshape(-1)
        highlight_map = cv2.blur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius),borderType = cv2.BORDER_DEFAULT)
        highlight_map = highlight_map.reshape(-1)
    # Tone LUT
    t = np.arange(256)
    LUT_shadow = (1 - np.power(1 - t * (1 / 255), shadow_gain)) * 255
    LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + .5)))
    LUT_highlight = np.power(t * (1 / 255), highlight_gain) * 255
    LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + .5)))

    # adjust tone
    shadow_map = shadow_map * (1 / 255)
    highlight_map = highlight_map * (1 / 255)

    iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
    iH = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]
    img_Y = iH

    # adjust color
    if color_percent != 0:
        # color LUT
        if color_percent > 0:
            LUT = (1 - np.sqrt(np.arange(32768)) * (1 / 128)) * color_percent + 1
        else:
            LUT = np.sqrt(np.arange(32768)) * (1 / 128) * color_percent + 1

        # adjust color saturation adaptively according to highlights/shadows
        color_gain = LUT[np.int_(img_U ** 2 + img_V ** 2 + .5)]
        w = 1 - np.minimum(2 - (shadow_map + highlight_map), 1)
        img_U = w * img_U + (1 - w) * img_U * color_gain
        img_V = w * img_V + (1 - w) * img_V * color_gain

    # re convert to RGB channel
    output_R = np.int_(img_Y + 1.402 * img_V + .5)
    output_G = np.int_(img_Y - .34414 * img_U - .71414 * img_V + .5)
    output_B = np.int_(img_Y + 1.772 * img_U + .5)

    output = np.row_stack([output_B, output_G, output_R]).T.reshape(height, width, 3)
    output = np.minimum(output, 255).astype(np.uint8)
    return output
from tensorflow.python.platform.tf_logging import error
def current_milli_time():
    return round(time.time() * 1000)
def fillhole(input_image):
    '''
    input gray binary image  get the filled image by floodfill method
    Note: only holes surrounded in the connected regions will be filled.
    :param input_image:
    :return:
    '''
    input_image*=255
    input_image=np.floor(input_image)
    input_image=input_image.astype("uint8")
    input_image[input_image>240]=255
    input_image[input_image<10]=0
    input_image = input_image.reshape(128,128)
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)    
    img_out = input_image | im_flood_fill_inv
    img_out=img_out.astype('float32')
    img_out/=255
    img_out = img_out.reshape(1,128,128,1)
    return img_out
def make_square(img):
    '''
    Reshapes the non-square image by pasting
    it to the centre of a black canvas of size
    n*n where n is the biggest dimension of
    the non-square image. 
    '''
    #Getting the bigger side of the image
    s = max(img.shape[0:2])

    #Creating a dark square with NUMPY  
    f = np.zeros((s,s,3),np.uint8)

    #Getting the centering position
    ax,ay = (s - img.shape[1])//2,(s - img.shape[0])//2

    #Pasting the 'image' in a centering position
    f[ay:img.shape[0]+ay,ax:ax+img.shape[1]] = img
    return f
def badfilter(imgin):
    tim = current_milli_time()
    pre = imgin
    pre = make_square(pre)
    pre = cv2.resize(pre, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    tam=correction(pre,1.,1.,0,0.,1.,0,0)
    tam=correction(tam,0.6,0.5,0,0.,0.4,0,0)
    tam = cv2.cvtColor(tam, cv2.COLOR_RGB2HSV)
    tam = cv2.fastNlMeansDenoisingColored(tam,None,10,6,7,21)

    pre = pre.astype('float32')
    pre /=255

    tam = tam.astype('float32')
    tam /=255
    #predict the mask 

    pred = model.predict(np.expand_dims(pre, 0))

    pred=fillhole(pred)

    #mask post-processing 
    hangsobien=6
    msk  = pred.squeeze()

    msk = cv2.copyMakeBorder(msk, hangsobien, hangsobien, hangsobien, hangsobien, cv2.BORDER_CONSTANT,value=[0,0,0])
    msk = cv2.resize(msk, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    msk = np.stack((msk,)*3, axis=-1)

    msk[msk >= 0.5] = 1 
    msk[msk < 0.5] = 0 
    M,N,k = msk.shape
    sum=0.0
    count=0
    hesoh=0
    hesos=3
    hangsos=0.45
    hesov=6
    hangsov=0.75
    hesotb=1.15
    tinhavglan2=0
    tinhavglan3=0

    for i in range(0,M):
        for j in range(0,N):
            if (msk[i,j,0]==1):
                tinhavglan2+=tam[i,j,1]
                tinhavglan3+=tam[i,j,2]      
                sum+=tam[i,j,0]*hesoh+(tam[i,j,1]/10+hangsos)*hesos+(tam[i,j,2]/10+hangsov)*hesov
                count+=1
    if (count!=0):
        avg=sum/count/hesotb
    else:
        avg = 0
    ermsk=np.zeros((128,128,3),np.uint8)
    tinh = 0
    nemsk = cv2.copyMakeBorder(msk, 2,2,2,2, cv2.BORDER_CONSTANT,value=[0,0,0])
    newtam = cv2.copyMakeBorder(tam, 2,2,2,2, cv2.BORDER_CONSTANT,value=[0,0,0])
    daucong = 2

    for i in range(daucong,M+daucong):
        for j in range(daucong,N+daucong):
            
            if ((nemsk[i,j,0]>0.8) & (newtam[i,j,0]*hesoh+newtam[i,j,1]*hesos+newtam[i,j,2]*hesov<avg)):
                lim = (daucong*2+1)*(daucong*2+1)-1
                xq = 0
                for a in range(-daucong,daucong+1):
                    for b in range(-daucong,daucong+1):
                        if (a,b != (0,0)):
                            if ((nemsk[i+a,j+b,0]>0.8) & (newtam[i+a,j+b,0]*hesoh+newtam[i+a,j+b,1]*hesos+newtam[i+a,j+b,2]*hesov<avg)):
                                xq+=1                
                #print(xq)
                if (lim!=0):
                    if (xq/lim > 0.5):
                        tinh+=1
                        ermsk[i-daucong,j-daucong,0]=1
                        ermsk[i-daucong,j-daucong,1]=1
                        ermsk[i-daucong,j-daucong,2]=1
                else:
                    tinh+=1
                    ermsk[i-daucong,j-daucong,0]=1
                    ermsk[i-daucong,j-daucong,1]=1
                    ermsk[i-daucong,j-daucong,2]=1
    #show the mask and the segmented image 
    print(current_milli_time()-tim)
    combined = np.concatenate([pre, msk, pre* msk,ermsk, pre*ermsk], axis = 1)
    combined = combined*255
    combined = combined.astype('uint8')
    if (count!=0):
        dapan=tinh/count
    else:
        dapan=0
    return combined, dapan
model = keras.models.load_model('model.h5', custom_objects={"mean_iou": mean_iou})
