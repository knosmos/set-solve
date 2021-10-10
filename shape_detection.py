import cv2
import numpy as np

WHITE_MIN = np.array([0, 0, 170],np.uint8)
WHITE_MAX = np.array([180, 60, 255],np.uint8)
AREA_THRESH = 7000

img = cv2.imread("cards/card10.png")
img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)

'''
# Get parts of card image that are not white
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_thresh = cv2.inRange(img_hsv, WHITE_MIN, WHITE_MAX)
img_thresh = cv2.bitwise_not(img_thresh)
cv2.imshow("image_thresh", img_thresh)

img_edge = cv2.Canny(img, 100, 200)
ret, img_thresh = cv2.threshold(img_edge, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("image_thresh", img_thresh)
'''

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray", img_gray)
ret, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("image_thresh", img_thresh)

# Get contours
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_copy1 = img.copy()
cv2.drawContours(img_copy1, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("contours", img_copy1)

# Remove contours that are inside others (contours-in-contours are caused by Canny)
shapes = []
for i, contour in enumerate(contours):
    print(f"hierarchy: {hierarchy[0,i,3]}")
    if hierarchy[0,i,3] == -1:
        print(cv2.contourArea(contour))
        shapes.append(contour)

'''
hulls = []
for contour in shapes:
    hulls.append(cv2.convexHull(contour, False))
'''

hulls = shapes # Temporary

'''
for i in range(len(hulls)):
    img_copy2 = img.copy()
    cv2.drawContours(img_copy2, [hulls[i]], -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("hull", img_copy2)
    cv2.waitKey()
'''

# Remove shapes that are too small
shapes = []
for hull in hulls:
    if cv2.contourArea(hull) >= AREA_THRESH:
        print(cv2.contourArea(hull))
        shapes.append(hull)

img_copy2 = img.copy()
cv2.drawContours(img_copy2, shapes, -1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("shapes", img_copy2)
print(f"shapes: {len(shapes)}")

cv2.waitKey()