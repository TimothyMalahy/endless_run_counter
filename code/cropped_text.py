import numpy as np
from PIL import Image, ImageFilter
import os


images = os.listdir(os.getcwd()+'\\sample_images\\cropped')

for image in images:
    im = Image.open(os.getcwd()+'\\sample_images\\cropped\\'+image).convert('RGB')
    na = np.array(im)
    orig = na.copy()
    # print(na)
    im = im.filter(ImageFilter.MedianFilter(3))
    
    whiteY, whiteX = np.where(np.all(na==[254,254,254], axis=2))
    
    top, bottom = whiteY[0], whiteY[-1]
    left, right = whiteX[0], whiteX[-1]
    print(top, bottom, left, right)
    
    ROI = orig[top:bottom, left:right]
    Image.fromarray(ROI).save(os.getcwd()+'\\sample_images\\double_cropped\\'+image)