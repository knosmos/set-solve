import cv2
import numpy as np
import sys

# Shape Separation Constants
WHITE_MIN = np.array([0, 0, 170],np.uint8)
WHITE_MAX = np.array([180, 60, 255],np.uint8)
AREA_THRESH = 7000

# Reference Features
OVAL_FEATURES = np.array([0.03155387787592858, -0.0037758243610366725, 0.205290481844915, -1.4503161754586032e-05, -2.9263679436010096e-05, 0.000142860777666745, 0.0001081177739358252])
DIAMOND_FEATURES = np.array([0.032902776926284946, -0.004827861419942702, 0.21169459653794936, -3.226527335538325e-05, 2.965936208089489e-06, 0.0001491130652801827, -6.661682099746633e-05])
SQUIGGLE_FEATURES = np.array([0.030580390546484034, 0.019392749469723414, 0.2646959376958161, 6.466367502274759e-05, 0.0005285519078694084, -0.0022415312093423457, -0.007887757425908665])

# Shading Constants
SOBEL_THRESH = 0.01
FILL_THRESH = 0.2

# Color Constants
COLOR_THRESH = 10
PURPLE_THRESH = 30

RED_BASELINE = np.array([45, 28, 234])
GREEN_BASELINE = np.array([80, 167, 20])
PURPLE_BASELINE = np.array([145, 45, 102])

def identify(img):
    # Shape Separation
    img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Get contours
    contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_copy1 = img.copy()

    # Remove contours that are inside others
    contours_filtered = []
    for i, contour in enumerate(contours):
        if hierarchy[0,i,3] == -1:
            contours_filtered.append(contour)
    contours = contours_filtered

    # Remove shapes that are too small
    contours_area_filtered = []
    for contour in contours:
        if cv2.contourArea(contour) >= AREA_THRESH:
            contours_area_filtered.append(contour)
    contours = contours_area_filtered

    img_copy2 = img.copy()
    cv2.drawContours(img_copy2, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)

    # Find number of shapes
    num_shapes = len(contours)
    if num_shapes > 3:
        print("scanning error - too many shapes; defaulting to 3")
        num_shapes = 3

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
    combined_distance = np.array([0.0,0.0,0.0])
    for feature in features:
        feature = np.array(feature)
        oval_dist = np.linalg.norm(feature - OVAL_FEATURES)
        diamond_dist = np.linalg.norm(feature - DIAMOND_FEATURES)
        squiggle_dist = np.linalg.norm(feature - SQUIGGLE_FEATURES)
        combined_distance += np.array([oval_dist, diamond_dist, squiggle_dist])

    min_dist = np.min(combined_distance)
    min_index = np.where(combined_distance == min_dist)[0][0]
    shape = ["oval", "diamond", "squiggle"][min_index]

    # Find Shading
    # Use SobelY to find stripes
    img_sobel = cv2.Sobel(img_thresh, cv2.CV_64F, 0, 1, ksize=1)

    sobel_white = np.sum(img_sobel == 255)/img_sobel.size
    thresh_white = np.sum(img_thresh == 255)/img_thresh.size

    # If there are a lot of white on the Sobel image, it's probably striped
    if sobel_white > SOBEL_THRESH/(4-num_shapes):
        shading = "striped"

    # If there is a lot of white on the thresh image, but not a lot
    # of white on the Sobel image, it's probably solid
    elif thresh_white > FILL_THRESH/(4-num_shapes):
        shading = "filled"

    # If there is not much white on either image, it's probably empty
    else:
        shading = "empty"

    # Find Color
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_thresh = cv2.inRange(img_hsv, WHITE_MIN, WHITE_MAX)
    img_thresh = cv2.bitwise_not(img_thresh)

    mean_color = np.array(cv2.mean(img, mask=img_thresh)[:-1])
    # print(mean_color)

    # Red
    if mean_color[2] > mean_color[1]+COLOR_THRESH and mean_color[2] > mean_color[0]+COLOR_THRESH:
        best_color = "red"
    # Purple
    elif mean_color[2] > mean_color[1]+COLOR_THRESH and mean_color[0]-mean_color[2] < PURPLE_THRESH:
        best_color = "purple"
    # Green
    elif mean_color[1] > mean_color[0]+COLOR_THRESH and mean_color[1] > mean_color[2]+COLOR_THRESH:
        best_color = "green"
    else:
        diffs = [
            np.linalg.norm(mean_color - RED_BASELINE),
            np.linalg.norm(mean_color - GREEN_BASELINE),
            np.linalg.norm(mean_color - PURPLE_BASELINE)
        ]
        #print(diffs)
        min_diff = min(diffs)
        min_index = diffs.index(min_diff)
        best_color = ["red", "green", "purple"][min_index]
    
    return {
        "num"    : num_shapes,
        "shape"  : shape,
        "shading": shading,
        "color"  : best_color,
    }

if __name__ == "__main__":
    print(identify(cv2.imread("cards/card5.png")))