import cv2
import numpy as np 

#Reading the original RSI image file
img_rgb = cv2.imread('o.png')
#Preprocessing the original image
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

#Loading the image for template matching
template = cv2.imread('t.png',0)

#Specifying the boundary dimensions
w, h = template.shape[::-1]

#Template matching with the original image
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.6
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

#Saving and displaying the image
cv2.imwrite('result.png',img_rgb)
cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
