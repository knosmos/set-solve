def test_set(t):
    # t: triplet of cards to test SET
    for feature in t[0].keys():
        # All the same or all different
        if not len(set([t[0][feature], t[1][feature], t[2][feature]])) in [1, 3]:
            return False
    return True

def solve(cards):
    num_cards = len(cards)
    solutions = []
    for i in range(num_cards):
        for j in range(i):
            for k in range(j):
                triplet = [cards[k], cards[j], cards[i]]
                if test_set(triplet):
                    solutions.append([k, j, i])
    return solutions