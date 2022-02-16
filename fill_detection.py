import cv2
import numpy as np

SOBEL_THRESH = 0.01
FILL_THRESH = 0.2

img = cv2.imread("cards/card10.png")
#img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("image_thresh", img_thresh)

# Use SobelY to find stripes
img_sobel = cv2.Sobel(img_thresh, cv2.CV_64F, 0, 1, ksize=1)
cv2.imshow("image_sobel", img_sobel)

sobel_white = np.sum(img_sobel == 255)/img_sobel.size
thresh_white = np.sum(img_thresh == 255)/img_thresh.size

print("sobel:  ", sobel_white)
print("thresh: ", thresh_white)

# If there are a lot of white on the Sobel image, it's probably striped
if sobel_white > SOBEL_THRESH:
    shading = "striped"

# If there is a lot of white on the thresh image, but not a lot
# of white on the Sobel image, it's probably solid
elif thresh_white > FILL_THRESH:
    shading = "solid"

# If there is not much white on either image, it's probably empty
else:
    shading = "empty"

print(shading)
cv2.waitKey()