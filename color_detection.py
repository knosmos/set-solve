import cv2
import numpy as np

# Shape Separation Constants
WHITE_MIN = np.array([0, 0, 170],np.uint8)
WHITE_MAX = np.array([180, 60, 255],np.uint8)

# Hue Constants
RED_HUE = 5.0
GREEN_HUE = 60.0
PURPLE_HUE = 125.0

img = cv2.imread("cards/card2.png")
img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
cv2.imshow("card", img)

img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_thresh = cv2.inRange(img_hsv, WHITE_MIN, WHITE_MAX)
img_thresh = cv2.bitwise_not(img_thresh)
cv2.imshow("thresh", img_thresh)

mean_hue = cv2.mean(img_hsv, mask=img_thresh)[0]

diffs = [abs(RED_HUE-mean_hue), abs(GREEN_HUE-mean_hue), abs(PURPLE_HUE-mean_hue)]
min_diff = min(diffs)
min_index = diffs.index(min_diff)
best_color = ["Red", "Green", "Purple"][min_index]

print("Average Hue:",mean_hue)
print("This card is probably", best_color)

cv2.waitKey()