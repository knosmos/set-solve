import cv2

min_thresh = 0
max_thresh = 255

def edge_detect():
    img_edge = cv2.Canny(img, min_thresh, max_thresh)
    ret, img_thresh = cv2.threshold(img_edge, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_copy1 = img.copy()
    cv2.drawContours(img_copy1, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("image", img_copy1)

def change_min(value):
    global min_thresh
    min_thresh = value
    print(f"min: {min_thresh} | max: {max_thresh}")
    edge_detect()

def change_max(value):
    global max_thresh
    max_thresh = value
    print(f"min: {min_thresh} | max: {max_thresh}")
    edge_detect()

img = cv2.imread('cards/card2.png')

cv2.imshow("image", img)
cv2.createTrackbar('min_thresh', 'image', 0, 255, change_min)
cv2.createTrackbar('max_thresh', 'image', 0, 255, change_max)

cv2.waitKey(0)
cv2.destroyAllWindows()