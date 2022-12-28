import cv2  
import numpy as np

import os

images = os.listdir(os.getcwd()+'\\sample_images')

images.remove('clears_icon.png')

print(images)

for image in images:
    image_file = os.getcwd()+'\\sample_images\\'+image
    orig_img = cv2.imread(image_file)
    # cv2.imshow("image_file"+str(image), orig_img)
    
    cropped_image = orig_img[60:140, 395:445]
    # print(image.split(".")[0])
    
    cv2.imwrite(os.getcwd()+'\\sample_images\\cropped\\'+image.split(".")[0]+"_cropped.png", cropped_image)
    
    
cv2.waitKey(0)
cv2.destroyAllWindows()