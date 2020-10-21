from UNet import *
import numpy as np
import imageio
from glob import glob
from skimage import img_as_float32 as as_float
from natsort import natsorted
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
import skimage.morphology as m
import cv2

path_pairs = list(zip(
natsorted(glob('./puzzle_corners_1024x768/images-1024x768/*.png')),
natsorted(glob('./puzzle_corners_1024x768/masks-1024x768/*.png')),
))

def ImportMask(mask):
    img = rgb2gray(mask)
    img = np.round(img) #make image binary
    img = m.opening(img, m.disk(3)) #fix noise in background
    return img

def K_fold(images, masks, k):
    length = images.shape[0]
    indices = np.arange(length)
    np.random.shuffle(indices)
    im = images[indices]
    ma = masks[indices]
    num_folds = np.round(length/k).astype(int)
    imageFolds = [im[i*num_folds:(i+1)*num_folds] for i in range(k)]
    maskFolds = [ma[i*num_folds:(i+1)*num_folds] for i in range(k)]
    return imageFolds, maskFolds

imgs = np.array([cv2.resize(cv2.resize(as_float(imageio.imread(ipath)), None, fx=0.75, fy=1),None,fx=0.25,fy=0.25) for ipath, _ in path_pairs])
msks = np.array([np.array(cv2.resize(cv2.resize(ImportMask(as_float(imageio.imread(mpath))), None, fx=0.75, fy=1),None,fx=0.25,fy=0.25)).reshape(192,192,1) for _, mpath in path_pairs])




# adaptive HE
# 