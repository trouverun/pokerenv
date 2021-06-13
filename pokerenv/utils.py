import numpy as np
from collections import Counter
from treys import Card

singulars = ['Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace']
plurals = ['Twos', 'Threes', 'Fours', 'Fives', 'Sixes', 'Sevens', 'Eights', 'Nines', 'Tens', 'Jacks', 'Queens', 'Kings', 'Aces']


def pretty_print_hand(hand_cards, hand_type, table_cards, kicker):
    combined = []
    combined.extend(hand_cards)
    combined.extend(table_cards)
    values = [Card.get_rank_int(c) for c in combined]
    suits = [Card.get_suit_int(c) for c in combined]
    suit_ints = [1, 2, 4, 8]
    if hand_type == 'High Card':
        return 'High card %s' % singulars[max(values)]
    if hand_type == 'Pair':
        doubles = []
        for k, v in Counter(values).items():
            doubles.extend([k] * (v // 2))
        return 'a pair of %s' % plurals[max(doubles)]
    if hand_type == 'Two Pair':
        doubles = []
        for k, v in Counter(values).items():
            doubles.extend([k] * (v // 2))
        first = max(doubles)
        doubles.remove(first)
        second = max(doubles)
        return 'two pair, %s and %s' % (plurals[first], plurals[second])
    if hand_type == 'Three of a Kind':
        triples = []
        for k, v in Counter(values).items():
            triples.extend([k] * (v // 3))
        return 'three of a kind, %s' % plurals[max(triples)]
    if hand_type == 'Straight':
        # TODO: Fix this mess
        previous_value = None
        high = None
        low = None
        sequence = 1
        for value in reversed(sorted(set(values))):
            if previous_value is None:
                previous_value = value
                high = value
                continue
            elif value == previous_value-1:
                sequence += 1
                if sequence == 4 and value == 0:
                    if 12 in values:
                        low = value
                        break
            else:
                sequence = 1
                high = value
            if sequence == 5:
                low = value
                break
            previous_value = value
        return 'a straight, %s to %s' % (singulars[low], singulars[high])
    if hand_type == 'Flush':
        # TODO: Fix this mess
        counts = np.array([len([suit for suit in suits if suit == i]) for i in suit_ints])
        suit_i = int(np.argmax(counts))
        high = max([values[i] for i in range(len(values)) if suits[i] == suit_ints[suit_i]])
        return 'a flush, %s high' % singulars[high]
    if hand_type == 'Full House':
        triples = []
        for k, v in Counter(values).items():
            triples.extend([k] * (v // 3))
        doubles = []
        for k, v in Counter(values).items():
            doubles.extend([k] * (v // 2))
        return 'a full house, %s full of %s' % (plurals[max(triples)], plurals[max(doubles)])
    if hand_type == 'Four of a Kind':
        quads = []
        for k, v in Counter(values).items():
            quads.extend([k] * (v // 4))
        return 'four of a kind, %s' % plurals[max(quads)]
    if hand_type == 'Straight Flush':
        # TODO: Fix this mess
        counts = np.array([len([suit for suit in suits if suit == i]) for i in suit_ints])
        suit_i = int(np.argmax(counts))
        correct_suit = [values[i] for i in range(len(values)) if suits[i] == suit_ints[suit_i]]

        previous_value = None
        high = None
        low = None
        sequence = 1
        for value in reversed(sorted(set(correct_suit))):
            if previous_value is None:
                previous_value = value
                high = value
                continue
            elif value == previous_value - 1:
                sequence += 1
                if sequence == 4 and value == 0:
                    if 12 in correct_suit:
                        low = value
                        break
            else:
                sequence = 1
                high = value
            if sequence == 5:
                low = value
                break
            previous_value = value
        return 'royal flush, %s to %s' % (singulars[low], singulars[high])
    raise Exception("Incorrect hand/table passed to pretty_print_hand")


def approx_lte(x, y):
    return x <= y or np.isclose(x, y)


def approx_gt(x, y):
    return x > y and not np.isclose(x, y)
