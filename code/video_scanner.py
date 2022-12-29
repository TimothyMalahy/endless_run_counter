import numpy
from time import time, sleep
import cv2 as cv2
import mss
from PIL import Image
import pytesseract
import numpy as np
from matplotlib import pyplot as plt


bounding_box = {'top': 46+111, 'left': 47, 'width': 1163, 'height': 622+38}

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

clears_0 = cv2.imread('sample_images/cropped/00_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_1 = cv2.imread('sample_images/cropped/01_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_2 = cv2.imread('sample_images/cropped/02_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_3 = cv2.imread('sample_images/cropped/03_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_4 = cv2.imread('sample_images/cropped/04_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_5 = cv2.imread('sample_images/cropped/05_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_6 = cv2.imread('sample_images/cropped/06_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_7 = cv2.imread('sample_images/cropped/07_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_8 = cv2.imread('sample_images/cropped/08_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_9 = cv2.imread('sample_images/cropped/09_clears_cropped.png', cv2.IMREAD_GRAYSCALE)
clears_icon = cv2.imread('sample_images/clears_icon.png', cv2.IMREAD_GRAYSCALE)
clears_icon_w = clears_icon.shape[1]
clears_icon_h = clears_icon.shape[0]

sct = mss.mss()

fps_time = time()
dimensions = {
    'left':47,
    'top':157,
    'width':1163,
    'height':660,
}

while True:
    scr = numpy.array(sct.grab(dimensions))
    scr_gray = cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY)
    # print(scr)
    # scr_remove = scr[:,:,:3]

    
    cv2.imshow('grayscale', cv2.cvtColor(scr, cv2.COLOR_BGR2GRAY))
    # cv2.imshow('clears_0', clears_0)
    # cv2.imshow('scr_rmove',scr_remove)

    result_0 = cv2.matchTemplate(scr_gray, clears_0, cv2.TM_CCOEFF_NORMED)
    result_1 = cv2.matchTemplate(scr_gray, clears_1, cv2.TM_CCOEFF_NORMED)
    result_2 = cv2.matchTemplate(scr_gray, clears_2, cv2.TM_CCOEFF_NORMED)
    result_3 = cv2.matchTemplate(scr_gray, clears_3, cv2.TM_CCOEFF_NORMED)
    result_4 = cv2.matchTemplate(scr_gray, clears_4, cv2.TM_CCOEFF_NORMED)
    result_5 = cv2.matchTemplate(scr_gray, clears_5, cv2.TM_CCOEFF_NORMED)
    result_6 = cv2.matchTemplate(scr_gray, clears_6, cv2.TM_CCOEFF_NORMED)
    result_7 = cv2.matchTemplate(scr_gray, clears_7, cv2.TM_CCOEFF_NORMED)
    result_8 = cv2.matchTemplate(scr_gray, clears_8, cv2.TM_CCOEFF_NORMED)
    result_9 = cv2.matchTemplate(scr_gray, clears_9, cv2.TM_CCOEFF_NORMED)
    result_clears_icon = cv2.matchTemplate(scr_gray, clears_icon, cv2.TM_CCOEFF_NORMED)
    print('result_0', cv2.minMaxLoc(result_0))
    print('result_1', cv2.minMaxLoc(result_1))
    print('result_2', cv2.minMaxLoc(result_2))
    print('result_3', cv2.minMaxLoc(result_3))
    print('result_4', cv2.minMaxLoc(result_4))
    print('result_5', cv2.minMaxLoc(result_5))
    print('result_6', cv2.minMaxLoc(result_6))
    print('result_7', cv2.minMaxLoc(result_7))
    print('result_8', cv2.minMaxLoc(result_8))
    print('result_9', cv2.minMaxLoc(result_9))
    print('result_clears_icon', cv2.minMaxLoc(result_clears_icon))
    
    
    threshold = .50

    yloc, xloc = np.where(result_clears_icon >= threshold)
    rectangles = []
    for (x, y) in zip(xloc, yloc):
        rectangles.append([int(x), int(y), int(clears_icon_w), int(clears_icon_h)])
        rectangles.append([int(x), int(y), int(clears_icon_w), int(clears_icon_h)])
        
    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

# print(len(rectangles))

    for (x, y, w, h) in rectangles:
        cv2.rectangle(scr, (x,y), (x+ w, y+h), (0,255,255), 2)
        
    cv2.imshow('',scr)
        
    # _, max_val, _, max_loc = cv2.minMaxLoc(result)
    # print(f"Max Val: {max_val} Max Loc: {max_loc}")
    # src = scr.copy()
    # if max_val > .85:
    
    # rectangles = []
    # for (x, y) in zip(xloc, yloc):
    #     rectangles.append([int(x), int(y), int(w), int(h)])
    #     rectangles.append([int(x), int(y), int(w), int(h)])

    # cv2.rectangle(scr, max_loc, (max_loc[0], max_loc[1]), (0,255,255), 2)
    # cv2.imshow('Screen shot', scr)
    if cv2.waitKey(1) == ord("q"):
        break
    
    

# img_rgb = cv2.imread('sample_images/01_clears.png')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# template = cv2.imread('sample_images/clears_icon.png', 0)
# w, h = template.shape[::-1]


# res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8
# loc = np.where( res>= threshold)
# for pt in zip(*loc[::-1]):
    # cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
# cv2.imwrite('res.png',img_rgb)


# with mss.mss() as sct:
#     # Part of the screen to capture
#     # monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
#     start_point = (190,40)
#     end_point = (190+55, 40+60)
#     color = (255, 0, 0)
#     thickness=2

#     while "Screen capturing":
#         last_time = time.time()

#         # Get raw pixels from the screen, save it to a Numpy array
#         frame = numpy.array(sct.grab(bounding_box))
#         frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         clears_icon = cv2.imread('sample_images/clears_icon.png')
#         clears_icon_gray = cv2.cvtColor(clears_icon, cv2.COLOR_BGR2GRAY)
               
#         cv2.imshow('clears_icon_gray', clears_icon_gray)
#         w = clears_icon_gray.shape[1]
#         h = clears_icon_gray.shape[0]
        
#         result = cv2.matchTemplate(clears_icon_gray, frame_gray, cv2.TM_CCOEFF_NORMED)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#         threshold = .80
        
#         yloc, xloc = np.where(result >= threshold)
#         for (x, y) in zip(xloc, yloc):
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,255), 2)
        
#         cv2.imshow('test',frame)
        
#         if cv2.waitKey(1) == ord("q"):
#             break
#         # Display the picture
        
#         # cv2.rectangle(img, start_point, end_point, color, thickness)
#         # res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
#         # loc = np.where( res>= threshold)
#         # for pt in zip(*loc[::-1]):
#             # cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
#         # cv2.imwrite('res.png',img_rgb)
#         # cv2.imshow("OpenCV/Numpy normal", frame)

#         # Display the picture in grayscale
#         # cv2.imshow('OpenCV/Numpy grayscale',
#                 #    cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY))

#         # print(f"fps: {1 / (time.time() - last_time)}")
#         # print(img)
#         # time.sleep(1)

#         # Press "q" to quit
#         if cv2.waitKey(25) & 0xFF == ord("q"):
#             cv2.destroyAllWindows()
#             break
# 0 clears - https://youtu.be/8Cl2NfYNlxk?t=7942 11 pixels
# 1 clears - https://youtu.be/8Cl2NfYNlxk?t=8030 17 pixels
# 2 clears - https://youtu.be/8Cl2NfYNlxk?t=8079
# 3 Clears - https://youtu.be/8Cl2NfYNlxk?t=8159
# 4 clears - https://youtu.be/8Cl2NfYNlxk?t=8613
# 5 clears - https://youtu.be/8Cl2NfYNlxk?t=8701
# 6 clears - https://youtu.be/8Cl2NfYNlxk?t=8818 
# 7 clears - https://youtu.be/8Cl2NfYNlxk?t=8878
# 8 clears - https://youtu.be/8Cl2NfYNlxk?t=9020
# 9 clears - https://youtu.be/8Cl2NfYNlxk?t=9147
# 10 clears - https://youtu.be/8Cl2NfYNlxk?t=9207
# 11  clears - https://youtu.be/8Cl2NfYNlxk?t=9277
# 12  clears - https://youtu.be/8Cl2NfYNlxk?t=9352
# 13  clears - https://youtu.be/8Cl2NfYNlxk?t=9412
# 14  clears - https://youtu.be/8Cl2NfYNlxk?t=9511
# 15  clears - https://youtu.be/8Cl2NfYNlxk?t=9569
# 16  clears - https://youtu.be/8Cl2NfYNlxk?t=9621
# 17  clears - https://youtu.be/8Cl2NfYNlxk?t=9717
# 18  clears - https://youtu.be/8Cl2NfYNlxk?t=9888
# 19  clears - https://youtu.be/8Cl2NfYNlxk?t=9959
# 20  clears - https://youtu.be/8Cl2NfYNlxk?t=10082