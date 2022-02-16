import card_separation
import card_identifier
import solver
import cv2
from rich import print

def extract(img):
    cards = card_separation.card_segment(img)
    res = []
    for card in cards:
        res.append(card_identifier.identify(card))
    return res

def solve(cards):
    return solver.solve(cards)

cards = extract(cv2.imread("test/set6.jpg"))
print(cards)
print(solve(cards))