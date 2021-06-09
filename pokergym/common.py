from enum import IntEnum, Enum


class GameState(IntEnum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


class PlayerState(Enum):
    FOLDED = 0
    ACTIVE = 1


class PlayerAction(IntEnum):
    CHECK = 0
    FOLD = 1
    BET = 2
    CALL = 3


class TablePosition(IntEnum):
    SB = 0
    BB = 1


class Action:
    def __init__(self, action_type, bet_amount=0):
        self.action_type = action_type
        self.bet_amount = bet_amount


action_list = [PlayerAction.CHECK, PlayerAction.FOLD, PlayerAction.BET, PlayerAction.CALL]
