import cv2
import numpy as np

# Shape Separation Constants
WHITE_MIN = np.array([0, 0, 170],np.uint8)
WHITE_MAX = np.array([180, 60, 255],np.uint8)
AREA_THRESH = 7000

# Hue Constants
RED_HUE = 5.0
GREEN_HUE = 60.0
PURPLE_HUE = 125.0

# Reference Features
ELLIPSE_FEATURES = np.array([0.03155387787592858, -0.0037758243610366725, 0.205290481844915, -1.4503161754586032e-05, -2.9263679436010096e-05, 0.000142860777666745, 0.0001081177739358252])
DIAMOND_FEATURES = np.array([0.032902776926284946, -0.004827861419942702, 0.21169459653794936, -3.226527335538325e-05, 2.965936208089489e-06, 0.0001491130652801827, -6.661682099746633e-05])
SQUIGGLE_FEATURES = np.array([0.030580390546484034, 0.019392749469723414, 0.2646959376958161, 6.466367502274759e-05, 0.0005285519078694084, -0.0022415312093423457, -0.007887757425908665])

FILLED_SAT = 110.0
STRIPE_SAT = 55.0
EMPTY_SAT = 0.0

# Shape Separation

img = cv2.imread("cards/card7.png")
img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow("image_gray", img_gray)
ret, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#cv2.imshow("image_thresh", img_thresh)

# Get contours
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img_copy1 = img.copy()
#cv2.drawContours(img_copy1, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("contours", img_copy1)

# Remove contours that are inside others
contours_filtered = []
for i, contour in enumerate(contours):
    #print(f"hierarchy: {hierarchy[0,i,3]}")
    if hierarchy[0,i,3] == -1:
        #print(cv2.contourArea(contour))
        contours_filtered.append(contour)
contours = contours_filtered

# Remove shapes that are too small
contours_area_filtered = []
for contour in contours:
    if cv2.contourArea(contour) >= AREA_THRESH:
        #print(cv2.contourArea(contour))
        contours_area_filtered.append(contour)
contours = contours_area_filtered

img_copy2 = img.copy()
cv2.drawContours(img_copy2, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("contours", img_copy2)

# Find number of shapes
print(f"Number: {len(contours)}")

# Find shape of shapes
features = []
for contour in contours:
    m = cv2.moments(contour)
    features.append([
        m["nu20"],
        m["nu11"],
        m["nu02"],
        m["nu30"],
        m["nu21"],
        m["nu12"],
        m["nu03"]
    ])
#print(features)
combined_distance = np.array([0.0,0.0,0.0])
for feature in features:
    feature = np.array(feature)
    ellipse_dist = np.linalg.norm(feature - ELLIPSE_FEATURES)
    diamond_dist = np.linalg.norm(feature - DIAMOND_FEATURES)
    squiggle_dist = np.linalg.norm(feature - SQUIGGLE_FEATURES)
    #print(f"ellipse: {ellipse_dist}, diamond: {diamond_dist}, squiggle: {squiggle_dist}")
    combined_distance += np.array([ellipse_dist, diamond_dist, squiggle_dist])

min_dist = np.min(combined_distance)
min_index = np.where(combined_distance == min_dist)[0][0]
shape = ["Ellipse", "Diamond", "Squiggle"][min_index]
print("Shape:",shape)

# Find Shading
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = np.zeros_like(img)
cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)
mask = cv2.inRange(mask, (250,250,250), (255,255,255))
#cv2.imshow("shading_mask", mask)
mean_shade = cv2.mean(img_hsv, mask=mask)[1]
print("average saturation:",mean_shade)

filled_dist = abs(mean_shade-FILLED_SAT)
stripe_dist = abs(mean_shade-STRIPE_SAT)
empty_dist = abs(mean_shade-EMPTY_SAT)
diffs = [filled_dist, stripe_dist, empty_dist]
min_diff = min(diffs)
min_index = diffs.index(min_diff)

best_shading = ["filled", "striped", "empty"][min_index]
print("Shading:", best_shading)

# Find Color
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_thresh = cv2.inRange(img_hsv, WHITE_MIN, WHITE_MAX)
img_thresh = cv2.bitwise_not(img_thresh)
#cv2.imshow("thresh", img_thresh)

mean_hue = cv2.mean(img_hsv, mask=img_thresh)[0]

diffs = [abs(RED_HUE-mean_hue), abs(GREEN_HUE-mean_hue), abs(PURPLE_HUE-mean_hue)]
min_diff = min(diffs)
min_index = diffs.index(min_diff)
best_color = ["red", "green", "purple"][min_index]

#print("Average Hue:",mean_hue)
print("Color:", best_color)

cv2.waitKey()