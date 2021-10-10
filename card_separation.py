import math
import cv2
import numpy as np

CARD_RATIO = 4.3/3.3 # dimensions from Amazon
CARD_IMG_WIDTH = 400
CARD_IMG_HEIGHT = int(CARD_IMG_WIDTH/CARD_RATIO)

# Read image
img = cv2.imread("test/set2.jpg")

# Extract white cards
WHITE_MIN = np.array([0, 0, 150],np.uint8)
WHITE_MAX = np.array([180, 100, 255],np.uint8)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_thresh = cv2.inRange(img_hsv, WHITE_MIN, WHITE_MAX)
cv2.imshow("image_thresh", img_thresh)

# Find Contours
contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_copy1 = img.copy()
cv2.drawContours(img_copy1, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
# cv2.imshow("contours", img_copy1)

# Find Cards
card_contours = []
for contour in contours:
    # Get rid of all the contours that are not four-sided
    perimeter = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
    if len(vertices) != 4:
        continue
    card_contours.append(vertices)

# We know there are exactly 12 cards, so find the 12 largest contours
card_contours.sort(key=cv2.contourArea, reverse=True)
card_contours = card_contours[:12]

# Sort
card_contours.sort(key=lambda k:k[0][0][1])
card_contours_sorted = []
for i in range(0,12,3):
    card_contours_sorted += sorted(card_contours[i:i+3], key=lambda k:k[0][0][0])
card_contours = card_contours_sorted

img_copy2 = img.copy()
cv2.drawContours(img_copy2, card_contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow("cards", img_copy2)

# Perspective-skew each card
vertices = np.float32([[0,CARD_IMG_HEIGHT], [0,0], [CARD_IMG_WIDTH,0], [CARD_IMG_WIDTH,CARD_IMG_HEIGHT]])
cards = []
for card in card_contours:
    # Reformat card contour because findContour gives strange results
    card = card.reshape(4,2)
    card = np.array(card, np.float32)

    # Orient points the right way
    lengths = []
    for i in range(3):
        p1, p2 = card[i], card[i-1]
        dist = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        lengths.append(dist)
    if (lengths[0] < lengths[1]):
        card = np.roll(card, 2)

    # Dewarp
    matrix = cv2.getPerspectiveTransform(card, vertices)
    card_dewarped = cv2.warpPerspective(img, matrix, (CARD_IMG_WIDTH, CARD_IMG_HEIGHT))
    card_dewarped = cv2.flip(card_dewarped, 1) # Temporary fix until I figure out what's flipping it
    cards.append(card_dewarped)

# Output
for i in range(len(cards)):
    cv2.imwrite(f"cards/card{i}.png", cards[i])

cv2.waitKey()