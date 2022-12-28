import cv2
import numpy as np

sample_image = cv2.imread('sample_images/01_clears.png', cv2.IMREAD_UNCHANGED)
clears_icon = cv2.imread('sample_images/clears_icon.png', cv2.IMREAD_UNCHANGED)

cv2.imshow('sample_image', sample_image)
# cv2.waitKey()
# cv2.destroyAllWindows()

# cv2.imshow('clears_icon', clears_icon)


result = cv2.matchTemplate(sample_image, clears_icon, cv2.TM_CCOEFF_NORMED)
# cv2.imshow('Result', result)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(max_loc)

w = clears_icon.shape[1]
h = clears_icon.shape[0]

# cv2.rectangle(sample_image, max_loc, (max_loc[0] + w, max_loc[1] + h), (0,255,255), 2)

threshold = .85

yloc, xloc = np.where(result >= threshold)
# print(len(yloc))
# print(len(xloc))

# for (x, y) in zip(xloc, yloc):
#     cv2.rectangle(sample_image, (x, y), (x+w, y+h), (0,255,255),2)
    
rectangles = []
for (x, y) in zip(xloc, yloc):
    rectangles.append([int(x), int(y), int(w), int(h)])
    rectangles.append([int(x), int(y), int(w), int(h)])
    
# print(len(rectangles))
    
rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

# print(len(rectangles))

for (x, y, w, h) in rectangles:
    cv2.rectangle(sample_image, (x,y), (x+ w, y+h), (0,255,255), 2)

cv2.imshow('',sample_image)

cv2.waitKey()
cv2.destroyAllWindows()