import numpy
import time
import cv2 as cv2
import mss
from PIL import Image
import pytesseract
import numpy as np
from matplotlib import pyplot as plt

bounding_box = {'top': 46+111, 'left': 47, 'width': 1163, 'height': 622+38}

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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


with mss.mss() as sct:
    # Part of the screen to capture
    # monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
    start_point = (190,40)
    end_point = (190+55, 40+60)
    color = (255, 0, 0)
    thickness=2

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        frame = numpy.array(sct.grab(bounding_box))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        clears_icon = cv2.imread('sample_images/clears_icon.png')
        clears_icon_gray = cv2.cvtColor(clears_icon, cv2.COLOR_BGR2GRAY)
               
        cv2.imshow('clears_icon_gray', clears_icon_gray)
        w = clears_icon_gray.shape[1]
        h = clears_icon_gray.shape[0]
        
        result = cv2.matchTemplate(clears_icon_gray, frame_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        threshold = .80
        
        yloc, xloc = np.where(result >= threshold)
        for (x, y) in zip(xloc, yloc):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,255), 2)
        
        cv2.imshow('test',frame)
        
        if cv2.waitKey(1) == ord("q"):
            break
        # Display the picture
        
        # cv2.rectangle(img, start_point, end_point, color, thickness)
        # res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        # loc = np.where( res>= threshold)
        # for pt in zip(*loc[::-1]):
            # cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        # cv2.imwrite('res.png',img_rgb)
        # cv2.imshow("OpenCV/Numpy normal", frame)

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
                #    cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY))

        # print(f"fps: {1 / (time.time() - last_time)}")
        # print(img)
        # time.sleep(1)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
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