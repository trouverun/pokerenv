ACTING_PLAYER = 0
HAND_ENDED = 1          # Hand ended this turn
ACTION_DONT_CARE = 2    # Hand ended before this turn (== actions are ignored and rewards are total winnings/losses on this hand)
VALID_ACTIONS = [*range(3, 7)]
VALID_BET_LOW = 7
VALID_BET_HIGH = 8
ACTING_PLAYER_POSITION = 9
ACTING_PLAYER_STACK_SIZE = 14
POT_SIZE = 22