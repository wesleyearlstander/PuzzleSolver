import numpy as np
import keras
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.backend import clear_session

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

#Setup tensorflow with GPU
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class UNet:
    '''
    UNet extending VGG16 using keras
    '''
    def __init__(self,img, learning_rate = 1e-4):
        img = np.array(img)
        self.input = Input(shape=(img.shape[0], img.shape[1], img.shape[2]))
        self.vgg16 = VGG16(input_tensor=self.input, include_top=False, weights = 'imagenet')
        self.vgg16.trainable = False
        self.c1 = self.vgg16.get_layer("block1_conv2").output 
        self.c2 = self.vgg16.get_layer("block2_conv2").output 
        self.c3 = self.vgg16.get_layer("block3_conv3").output  
        self.c4 = self.vgg16.get_layer("block4_conv3").output 
        self.c5 = self.vgg16.get_layer("block5_conv3").output
        self.c6 = self.AddUpsampleLayer(self.c5, self.c4, 512, True) 
        self.c7 = self.AddUpsampleLayer(self.c6, self.c3, 256, True)
        self.c8 = self.AddUpsampleLayer(self.c7, self.c2, 128)
        self.c9 = self.AddUpsampleLayer(self.c8, self.c1, 64)
        self.outputs = Conv2D(1, (1, 1), activation='sigmoid')(self.c9)
        self.UNet = Model(inputs=self.input, outputs=self.outputs, name="UNet")
        self.UNet.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy', metrics = ['accuracy'])
        

    def AddUpsampleLayer(self, inLayer, concatLayer, inputSize, triple=False):
        u = Conv2D(inputSize, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(inLayer))    
        u = concatenate([u, concatLayer], axis=3)
        u = Conv2D(inputSize, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u)
        u = Conv2D(inputSize, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u)
        if triple:
            u = Conv2D(inputSize, 3, activation='relu', kernel_initializer='he_normal', padding='same')(u)
        return u
