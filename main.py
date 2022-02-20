import card_separation
import card_identifier
import solver
import cv2
from rich import print
from flask import Flask, render_template, request

def extract(img):
    cards = card_separation.card_segment(img)
    res = []
    for card in cards:
        res.append(card_identifier.identify(card))
    return res

def solve(cards):
    return solver.solve(cards)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("request.files:", request.files)
        file = request.files['file']
        if file:
            file.save("tmp.png")
            img = cv2.imread("tmp.png")
            cards = extract(img)
            solutions = solve(cards)
            print(cards)
            print(solutions)
            return render_template('display.html', cards=cards, solutions=solutions)
    return render_template('index.html')

'''
cards = extract(cv2.imread("test/set9.jpg"))
print(cards)
print(solve(cards))
'''

if __name__ == "__main__":
    app.run("0.0.0.0", debug=True)