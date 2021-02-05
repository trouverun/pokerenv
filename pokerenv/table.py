import numpy as np
import time
from treys import Deck, Evaluator, Card
from pokerenv.common import GameState, PlayerState, PlayerAction, TablePosition
from pokerenv.player import Player
from pokerenv.utils import pretty_print_hand

# Just some values to make hand history work properly
SB = 2.5
BB = 5


class Table:
    def __init__(self, n_players, agents, seed, stack_low=50, stack_high=200, hand_history_location='hands/', invalid_action_penalty=-5):
        self.hand_history_location = hand_history_location
        self.hand_history_enabled = False
        self.stack_low = stack_low
        self.stack_high = stack_high
        self.rng = np.random.default_rng(seed)
        self.n_players = n_players
        self.pot = 0
        self.bet_to_match = 0
        self.minimum_raise = 0
        self.street = GameState.PREFLOP
        self.cards = []
        self.deck = Deck()
        self.players = [Player(n+1, agents[n], 'player_%d' % n, invalid_action_penalty) for n in range(n_players)]
        self.active_players = n_players
        self.evaluator = Evaluator()
        self.history = []

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.pot = 0
        self.street = GameState.PREFLOP
        self.cards = []
        self.deck.cards = Deck.GetFullDeck()
        self.rng.shuffle(self.deck.cards)
        self.rng.shuffle(self.players)
        self.active_players = self.n_players
        initial_draw = self.deck.draw(self.n_players * 2)
        for i, player in enumerate(self.players):
            player.reset()
            player.position = i
            player.cards = [initial_draw[i], initial_draw[i+self.n_players]]
            player.stack = self.rng.integers(self.stack_low, self.stack_high, 1)[0]
        self.bet_to_match = 0
        self.history = []

    def play_hand(self):
        if self.hand_history_enabled:
            self._history_initialize()

        blinds_collected = False
        hand_is_over = False
        for street in GameState:
            street_finished = False
            last_bet_placed_by = None
            self.street = street
            self.bet_to_match = 0
            self.minimum_raise = 0

            for player in self.players:
                player.finish_street()

            if street == GameState.FLOP:
                self.cards = self.deck.draw(3)
                self._write_event("*** FLOP *** [%s %s %s]" %
                                    (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                                     Card.int_to_str(self.cards[2])))
            if street == GameState.TURN:
                new = self.deck.draw(1)
                self.cards.append(new)
                self._write_event("*** TURN *** [%s %s %s] [%s]" %
                                    (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                                     Card.int_to_str(self.cards[2]), Card.int_to_str(self.cards[3])))
            if street == GameState.RIVER:
                new = self.deck.draw(1)
                self.cards.append(new)
                self._write_event("*** RIVER *** [%s %s %s %s] [%s]" %
                                    (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                                     Card.int_to_str(self.cards[2]), Card.int_to_str(self.cards[3]),
                                     Card.int_to_str(self.cards[4])))

            while not street_finished and not hand_is_over:
                for player in self.players:
                    if player.all_in or player.state is not PlayerState.ACTIVE:
                        continue

                    if street == GameState.PREFLOP and not blinds_collected:
                        if player.position == TablePosition.SB:
                            self.pot += player.bet(0.5)
                            self._change_bet_to_match(0.5)
                            self._write_event("%s: posts small blind $%.2f" % (player.name, SB))
                            continue
                        elif player.position == TablePosition.BB:
                            self.pot += player.bet(1)
                            self._change_bet_to_match(1)
                            last_bet_placed_by = player
                            blinds_collected = True
                            self._write_event("%s: posts big blind $%.2f" % (player.name, BB))
                            if self.hand_history_enabled:
                                self._write_hole_cards()
                            continue

                    have_actions = [p for p in self.players if p.state is PlayerState.ACTIVE if not p.all_in]
                    if len(have_actions) < 2:
                        amount = 0
                        # Everone else is all in or folded
                        if self.active_players > 1:
                            street_finished = True
                            biggest_match = max([p.bet_this_street for p in self.players if p.state is PlayerState.ACTIVE if p is not player])
                            if biggest_match < player.bet_this_street:
                                amount = player.bet_this_street - biggest_match
                        # Everone else has folded
                        else:
                            hand_is_over = True
                            amount = self.minimum_raise
                        if amount > 0:
                            self.pot -= amount
                            player.stack += amount
                            player.money_in_pot -= amount
                            player.bet_this_street -= amount
                            self._write_event(
                                    "Uncalled bet ($%.2f) returned to %s" % (amount * BB, player.name)
                            )
                        break

                    # If no one has raised after a bet, the betting round is over
                    if last_bet_placed_by is player:
                        street_finished = True
                        break

                    observation = self._get_observation(player)
                    valid_actions = self._get_valid_actions(player)
                    action = player.step(observation, valid_actions)
                    # If action is not valid, a valid action (check or fold) is automatically taken in _is_action_valid
                    if not self._is_action_valid(player, action, valid_actions):
                        player.punish_invalid_action()
                        continue

                    if action.action_type is PlayerAction.FOLD:
                        player.fold()
                        self.active_players -= 1
                        self._write_event("%s: folds" % player.name)
                    elif action.action_type is PlayerAction.CHECK:
                        player.check()
                        self._write_event("%s: checks" % player.name)
                    elif action.action_type is PlayerAction.CALL:
                        call_size = player.call(self.bet_to_match)
                        self.pot += call_size
                        if player.all_in:
                            self._write_event("%s: calls $%.2f and is all-in" % (player.name, call_size*BB))
                        else:
                            self._write_event("%s: calls $%.2f" % (player.name, call_size*BB))
                    elif action.action_type is PlayerAction.BET:
                        previous_bet_this_street = player.bet_this_street
                        actual_bet_size = player.bet(np.round(action.bet_amount, 2))
                        self.pot += actual_bet_size
                        if self.bet_to_match == 0:
                            if player.all_in:
                                self._write_event("%s: bets $%.2f and is all-in" % (player.name, actual_bet_size*BB))
                            else:
                                self._write_event("%s: bets $%.2f" % (player.name, actual_bet_size*BB))
                        else:
                            if player.all_in:
                                self._write_event("%s: raises $%.2f to $%.2f and is all-in" %
                                                  (player.name,
                                                   ((actual_bet_size+previous_bet_this_street)-self.bet_to_match)*BB,
                                                   (actual_bet_size+previous_bet_this_street)*BB)
                                                  )
                            else:
                                self._write_event("%s: raises $%.2f to $%.2f" %
                                                  (player.name,
                                                   ((actual_bet_size+previous_bet_this_street)-self.bet_to_match)*BB,
                                                   (actual_bet_size+previous_bet_this_street)*BB)
                                                  )
                        self._change_bet_to_match(actual_bet_size+previous_bet_this_street)
                        last_bet_placed_by = player
                    else:
                        raise Exception('Invalid action specified')
                if last_bet_placed_by is None:
                    break
            if hand_is_over:
                break
        if not hand_is_over:
            if self.hand_history_enabled:
                self._write_show_down()
        self._distribute_pot()
        self._finish_hand()

    def _history_initialize(self):
        t = time.localtime()
        self.history.append("PokerStars Hand #%d: Hold'em No Limit ($%.2f/$%.2f USD) - %d/%d/%d %d:%d:%d ET" %
                            (np.random.randint(2230397, 32303976), SB, BB, t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour,
                             t.tm_min, t.tm_sec))
        self.history.append("Table 'Wempe III' 6-max Seat #%d is the button" % self.players[min(1, 2)].identifier)
        for i, player in enumerate(self.players):
            self.history.append("Seat %d: %s ($%.2f in chips)" % (i+1, player.name, player.stack*BB))

    def _write_hole_cards(self):
        self.history.append("*** HOLE CARDS ***")
        for i, player in enumerate(self.players):
            self.history.append("Dealt to %s [%s %s]" %
                                (player.name, Card.int_to_str(player.cards[0]), Card.int_to_str(player.cards[1])))

    def _write_show_down(self):
        self.history.append("*** SHOW DOWN ***")
        hand_types = [self.evaluator.class_to_string(self.evaluator.get_rank_class(p.hand_rank))
                        for p in self.players if p.state is PlayerState.ACTIVE]
        for player in self.players:
            if player.state is PlayerState.ACTIVE:
                player.calculate_hand_rank(self.evaluator, self.cards)
                player_hand_type = self.evaluator.class_to_string(self.evaluator.get_rank_class(player.hand_rank))
                matches = len([m for m in hand_types if m is player_hand_type])
                multiple = matches > 1
                self.history.append("%s: shows [%s %s] (%s)" %
                                    (player.name, Card.int_to_str(player.cards[0]), Card.int_to_str(player.cards[1]),
                                     pretty_print_hand(player.cards, player_hand_type, self.cards, multiple))
                                    )

    def _write_event(self, text):
        if self.hand_history_enabled:
            self.history.append(text)

    def _change_bet_to_match(self, new_amount):
        self.minimum_raise = new_amount - self.bet_to_match
        self.bet_to_match = new_amount

    def _get_observation(self, player):
        keys = ['position', 'state', 'stack', 'money_in_pot', 'bet_this_street', 'all_in']
        values = [
            [other.position, other.state, other.stack, other.money_in_pot, other.bet_this_street, other.all_in]
            for other in self.players if other is not player
        ]
        return {
            'self': {
                'position': player.position,
                'cards': player.cards,
                'stack': player.stack,
                'money_in_pot': player.money_in_pot,
                'bet_this_street': player.bet_this_street,
            },
            'table': {
                'street': int(self.street),
                'cards': self.cards,
                'pot': self.pot,
                'bet_to_match': self.bet_to_match,
                'minimum_raise': self.minimum_raise,
            },
            'others': [
                {
                    key: val
                    for key, val in zip(keys, value)
                }
                for value in values
            ]

        }

    def _is_action_valid(self, player, action, valid_actions):
        action_list, bet_range = valid_actions['actions_list'], valid_actions['bet_range']
        if action.action_type not in action_list:
            if PlayerAction.FOLD in action_list:
                player.fold()
                self.active_players -= 1
                self._write_event("%s: folds" % player.name)
                return False
            if PlayerAction.CHECK in action_list:
                player.check()
                self._write_event("%s: checks" % player.name)
                return False
            raise Exception('Something went wrong when validating actions, invalid contents of valid_actions')
        if action.action_type is PlayerAction.BET:
            if not bet_range[0] < action.bet_amount < bet_range[1]:
                player.fold()
                self.active_players -= 1
                return False
            if action.bet_amount > player.stack:
                player.fold()
                self.active_players -= 1
                return False
        return True

    def _get_valid_actions(self, player):
        valid_actions = [PlayerAction.CHECK, PlayerAction.FOLD, PlayerAction.BET, PlayerAction.CALL]
        valid_bet_range = [max(self.bet_to_match + self.minimum_raise, 1), player.stack]
        if self.bet_to_match == 0:
            valid_actions.remove(PlayerAction.CALL)
            valid_actions.remove(PlayerAction.FOLD)
        if self.bet_to_match != 0:
            valid_actions.remove(PlayerAction.CHECK)
            if player.stack < max(self.bet_to_match + self.minimum_raise, 1):
                valid_bet_range = [0, 0]
                valid_actions.remove(PlayerAction.BET)
        return {'actions_list': valid_actions, 'bet_range': valid_bet_range}

    def _finish_hand(self):
        for player in self.players:
            player.step(None, None, True)
            if self.hand_history_enabled:
                if player.winnings > 0:
                    player.winnings = np.round(player.winnings, 2)
                    self._write_event("%s collected $%.2f from pot" % (player.name, player.winnings*BB))

        self._write_event("*** SUMMARY ***")
        self._write_event("Total pot $%.2f | Rake $%.2f" % (self.pot*BB, 0))
        if self.street == GameState.FLOP:
            self._write_event("Board [%s %s %s]" %
                                (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                                 Card.int_to_str(self.cards[2]))
                                )
        elif self.street == GameState.TURN:
            self._write_event("Board [%s %s %s %s]" %
                                (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                                 Card.int_to_str(self.cards[2]), Card.int_to_str(self.cards[3]))
                                )
        elif self.street == GameState.RIVER:
            self._write_event("Board [%s %s %s %s %s]" %
                                (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                                 Card.int_to_str(self.cards[2]), Card.int_to_str(self.cards[3]),
                                 Card.int_to_str(self.cards[4]))
                                )

        if self.hand_history_enabled and self.hand_history_location is not None:
            with open('%s/handhistory_%s.txt' % (self.hand_history_location,time.time()), 'w') as f:
                for row in self.history:
                    f.writelines(row + '\n')

    def _distribute_pot(self):
        pot = 0
        for player in self.players:
            if player.state is not PlayerState.ACTIVE:
                pot += player.money_in_pot
                player.winnings -= player.money_in_pot
        active_players = [p for p in self.players if p.state is PlayerState.ACTIVE]
        if len(active_players) == 1:
            active_players[0].winnings += pot + active_players[0].money_in_pot
            return
        for player in active_players:
            player.calculate_hand_rank(self.evaluator, self.cards)
        while True:
            min_money_in_pot = min([p.money_in_pot for p in active_players])
            for player in active_players:
                pot += min_money_in_pot
                player.money_in_pot -= min_money_in_pot
            best_hand_rank = min([p.hand_rank for p in active_players])
            winners = [p for p in active_players if p.hand_rank == best_hand_rank]
            for winner in winners:
                winner.winnings += pot / len(winners)
            active_players = [p for p in active_players if p.money_in_pot > 0]
            if len(active_players) <= 1:
                if len(active_players) == 1:
                    active_players[0].winnings += active_players[0].money_in_pot
                break
            pot = 0
