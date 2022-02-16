import cv2
import numpy as np

# Shape Separation

img = cv2.imread("cards/card4.png")
#img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("image_thresh", img_thresh)

# Use SobelY to find stripes
img_sobel = cv2.Sobel(img_thresh, cv2.CV_64F, 0, 1, ksize=1)
cv2.imshow("image_sobel", img_sobel)

cv2.waitKey()