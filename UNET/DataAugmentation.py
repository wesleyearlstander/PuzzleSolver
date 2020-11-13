from UNet import *
import numpy as np
import imageio
from glob import glob
from skimage import img_as_float32 as as_float
from natsort import natsorted
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv, hsv2rgb, rgb2gray
import skimage.morphology as m
from keras.preprocessing.image import ImageDataGenerator
import cv2

path_pairs = list(zip(
natsorted(glob('./puzzle_corners_1024x768/images-1024x768/*.png')),
natsorted(glob('./puzzle_corners_1024x768/masks-1024x768/*.png')),
))

def ImportMask(mask):
    img = rgb2gray(mask)
    img = np.round(img) #make image binary
    img = m.opening(img, m.disk(10)) #fix noise in background
    return img

def K_fold(images, masks, k): #Seperate data into k folds
    length = images.shape[0]
    indices = np.arange(length)
    np.random.shuffle(indices)
    im = images[indices]
    ma = masks[indices]
    num_folds = np.round(length/k).astype(int)
    imageFolds = [im[i*num_folds:(i+1)*num_folds] for i in range(k)]
    maskFolds = [ma[i*num_folds:(i+1)*num_folds] for i in range(k)]
    return imageFolds, maskFolds

def trainGenerator(trainImages,trainMasks,aug_dict,batch_size=2,seed = 1): 
    '''
    Augment data into training generator
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_datagen.fit(trainImages, augment=True, seed=seed)
    mask_datagen.fit(trainMasks, augment=True, seed=seed)
    image_generator = image_datagen.flow(
        trainImages,
        batch_size=batch_size,
        seed = seed)
    mask_generator = mask_datagen.flow(
        trainMasks,
        batch_size=batch_size,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    return train_generator

imgs = np.array([cv2.resize(cv2.resize(as_float(imageio.imread(ipath)), None, fx=0.75, fy=1),None,fx=0.25,fy=0.25) for ipath, _ in path_pairs])
msksCleaned = np.array([np.array(cv2.resize(cv2.resize(ImportMask(as_float(imageio.imread(mpath))), None, fx=0.75, fy=1),None,fx=0.25,fy=0.25)).reshape(192,192,1) for _, mpath in path_pairs])
msks = np.array([np.array(cv2.resize(cv2.resize(as_float(imageio.imread(mpath)), None, fx=0.75, fy=1),None,fx=0.25,fy=0.25)).reshape(192,192,1) for _, mpath in path_pairs])

msks[msks >= 0.5] = 1
msks[msks < 0.5] = 0
msksCleaned[msksCleaned >= 0.5] = 1
msksCleaned[msksCleaned < 0.5] = 0
