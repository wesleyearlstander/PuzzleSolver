from UNet import *
import numpy as np
import imageio
from glob import glob
from skimage import img_as_float32 as as_float
from natsort import natsorted
import matplotlib.pyplot as plt
import cv2

path_pairs = list(zip(
natsorted(glob('./puzzle_corners_1024x768/images-1024x768/*.png')),
natsorted(glob('./puzzle_corners_1024x768/masks-1024x768/*.png')),
))
imgs = np.array([cv2.resize(cv2.resize(as_float(imageio.imread(ipath)), None, fx=0.75, fy=1),None,fx=0.25,fy=0.25) for ipath, _ in path_pairs])
msks = np.array([as_float(imageio.imread(mpath)) for _, mpath in path_pairs])

model = UNet(imgs[0])
#model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.UNet.load_weights("unet.hdf5")
filters, biases = model.UNet.layers[1].get_weights()
fig = plt.figure(figsize=(5,10))
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
n_filters, ix = 6, 1
fig.tight_layout()
for i in range(n_filters):
	f = filters[:, :, :, i]
	for j in range(3):
		ax = plt.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		plt.imshow(f[:, :, j], cmap='gray')
		ix += 1
plt.show()