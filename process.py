import os
from PIL import Image
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import cv2
import numpy as np



def process(filename):
    path = os.path.join('/home/syanru/Solar-Wind-peed/', filename)
    im = Image.open(path)
    img = im.convert("RGB")
    return img


def ProcessImg(filename):
    '''
	The pre-preprocessing routine for the new images from LMSAL. These images have been sourced from
	/bigdata/FDL/AIA/211/*, on learning.lmsal_gpu.internals.
	The images need to be first log-scaled, and low-clipped at log(10). High clipping at log(10000).
	Finally, rescaling from 0 to 256.
	'''
    # First check if the image shape is correct.
    img_size = 256
    tv1 = 500
    tv2 = 20000

    path = os.path.join('/home/syanru/CMP-Solar-Wind-peed/', filename)
    x = Image.open(path)
    shp = x.size[0]
    if shp != 256:
        x = cv2.resize(np.array(x), (img_size, img_size))
    return x