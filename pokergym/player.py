from pokergym.common import PlayerState, PlayerAction


class Player:
    def __init__(self, identifier, name, penalty):
        self.state = PlayerState.ACTIVE
        self.has_acted = False
        self.acted_this_street = False
        self.identifier = identifier
        self.name = name
        self.stack = 0
        self.cards = []
        self.position = 0
        self.all_in = False
        self.bet_this_street = 0
        self.money_in_pot = 0
        self.history = []
        self.hand_rank = 0
        self.pending_penalty = 0
        self.winnings = 0
        self.winnings_for_hh = 0
        self.penalty = penalty

    def __lt__(self, other):
        return self.identifier < other.identifier

    def __gt__(self, other):
        return self.identifier > other.identifier

    def get_reward(self):
        if self.has_acted:
            tmp = self.pending_penalty
            self.pending_penalty = 0
            return tmp + self.winnings
        else:
            return None

    def fold(self):
        self.has_acted = True
        self.acted_this_street = True
        self.state = PlayerState.FOLDED
        self.history.append({'action': PlayerAction.FOLD, 'value': 0})

    def check(self):
        self.has_acted = True
        self.acted_this_street = True
        self.history.append({'action': PlayerAction.CHECK, 'value': 0})

    def call(self, amount):
        self.has_acted = True
        self.acted_this_street = True
        amount = amount - self.bet_this_street
        if amount >= self.stack:
            call_size = self.stack
            self.stack = 0
            self.all_in = True
            self.bet_this_street += call_size
            self.money_in_pot += call_size
            self.history.append({'action': PlayerAction.CALL, 'value': call_size})
            return call_size
        else:
            self.stack -= amount
            self.bet_this_street += amount
            self.money_in_pot += amount
            self.history.append({'action': PlayerAction.CALL, 'value': amount})
            return amount

    def bet(self, amount):
        self.has_acted = True
        self.acted_this_street = True
        if amount == self.stack:
            self.all_in = True
        amount = amount - self.bet_this_street
        self.stack -= amount
        self.bet_this_street += amount
        self.money_in_pot += amount
        self.history.append({'action': PlayerAction.BET, 'value': amount})
        return amount

    def punish_invalid_action(self):
        self.pending_penalty += self.penalty

    def finish_street(self):
        self.acted_this_street = False
        self.bet_this_street = 0

    def calculate_hand_rank(self, evaluator, community_cards):
        self.hand_rank = evaluator.evaluate(self.cards, community_cards)

    def reset(self):
        self.state = PlayerState.ACTIVE
        self.has_acted = False
        self.all_in = False
        self.bet_this_street = 0
        self.money_in_pot = 0
        self.cards = []
        self.history = []
        self.hand_rank = 0
        self.pending_penalty = 0
        self.winnings = 0
        self.winnings_for_hh = 0
