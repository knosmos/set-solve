import cv2
import numpy as np

# Shape Separation

img = cv2.imread("cards/card1.png")
#img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray", img_gray)

# Use SobelY to find stripes
img_sobel = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=1)
cv2.imshow("image_sobel", img_sobel)

cv2.waitKey()