import numpy as np
import time
import gym
import math
from treys import Deck, Evaluator, Card
from pokergym.common import GameState, PlayerState, PlayerAction, TablePosition, Action
from pokergym.player import Player
from pokergym.utils import pretty_print_hand, approx_gt, approx_lte

# Just some values to make hand history work properly
SB = 2.5
BB = 5


class Table(gym.Env):
    def __init__(self, n_players, stack_low=50, stack_high=200, hand_history_location='hands/', invalid_action_penalty=0):
        self.action_space = gym.spaces.Tuple((gym.spaces.Discrete(4), gym.spaces.Box(-math.inf, math.inf, (1, 1))))
        self.observation_space = gym.spaces.Box(-math.inf, math.inf, (60, 1))
        self.n_players = n_players
        self.all_players = [Player(n, 'player_%d' % (n+1), invalid_action_penalty) for n in range(n_players)]
        self.players = self.all_players[:n_players]
        self.active_players = n_players
        self.next_player_i = min(self.n_players-1, 2)
        self.current_player_i = self.next_player_i
        self.hand_history_location = hand_history_location
        self.hand_history_enabled = False
        self.hand_history = []
        self.stack_low = stack_low
        self.stack_high = stack_high
        self.current_turn = 0
        self.pot = 0
        self.bet_to_match = 0
        self.minimum_raise = 0
        self.street = GameState.PREFLOP
        self.deck = Deck()
        self.evaluator = Evaluator()
        self.cards = []
        self.rng = np.random.default_rng(None)
        self.street_finished = False
        self.hand_is_over = False
        self.hand_ended_last_turn = False
        self.last_bet_placed_by = None
        self.first_to_act = None
        self.final_rewards_collected = 0

    def seed(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.current_turn = 0
        self.pot = 0
        self.street = GameState.PREFLOP
        self.deck.cards = Deck.GetFullDeck()
        self.rng.shuffle(self.deck.cards)
        self.cards = []
        self.active_players = self.n_players
        self.players = self.all_players[:self.n_players]
        self.rng.shuffle(self.players)
        self.next_player_i = 0 if self.n_players == 2 else 2
        self.current_player_i = self.next_player_i
        self.first_to_act = None
        self.street_finished = False
        self.hand_is_over = False
        self.hand_ended_last_turn = False
        self.final_rewards_collected = 0
        initial_draw = self.deck.draw(self.n_players * 2)
        for i, player in enumerate(self.players):
            player.reset()
            player.position = i
            player.cards = [initial_draw[i], initial_draw[i+self.n_players]]
            player.stack = self.rng.integers(self.stack_low, self.stack_high, 1)[0]
        self.hand_history = []
        if self.hand_history_enabled:
            self._history_initialize()
        for i, player in enumerate(self.players):
            if player.position == TablePosition.SB:
                self.pot += player.bet(0.5)
                self._change_bet_to_match(0.5)
                self._write_event("%s: posts small blind $%.2f" % (player.name, SB))
            elif player.position == TablePosition.BB:
                self.pot += player.bet(1)
                self._change_bet_to_match(1)
                self.last_bet_placed_by = player
                self._write_event("%s: posts big blind $%.2f" % (player.name, BB))
        if self.hand_history_enabled:
            self._write_hole_cards()
        return self._get_observation(self.players[self.next_player_i])

    def step(self, action: Action):
        self.current_player_i = self.next_player_i
        player = self.players[self.current_player_i]
        self.current_turn += 1

        if (player.all_in or player.state is not PlayerState.ACTIVE) and not self.hand_is_over:
            raise Exception("A player who is inactive or all-in was allowed to act")
        if self.first_to_act is None:
            self.first_to_act = player

        # Apply the player action
        if not (self.hand_is_over or self.street_finished):
            valid_actions = self._get_valid_actions(player)
            if not self._is_action_valid(player, action, valid_actions):
                player.punish_invalid_action()
            elif action.action_type is PlayerAction.FOLD:
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
                    self._write_event("%s: calls $%.2f and is all-in" % (player.name, call_size * BB))
                else:
                    self._write_event("%s: calls $%.2f" % (player.name, call_size * BB))
            elif action.action_type is PlayerAction.BET:
                previous_bet_this_street = player.bet_this_street
                actual_bet_size = player.bet(np.round(action.bet_amount, 2))
                self.pot += actual_bet_size
                if self.bet_to_match == 0:
                    if player.all_in:
                        self._write_event("%s: bets $%.2f and is all-in" % (player.name, actual_bet_size * BB))
                    else:
                        self._write_event("%s: bets $%.2f" % (player.name, actual_bet_size * BB))
                else:
                    if player.all_in:
                        self._write_event("%s: raises $%.2f to $%.2f and is all-in" %
                                          (player.name,
                                           ((actual_bet_size + previous_bet_this_street) - self.bet_to_match) * BB,
                                           (actual_bet_size + previous_bet_this_street) * BB)
                                          )
                    else:
                        self._write_event("%s: raises $%.2f to $%.2f" %
                                          (player.name,
                                           ((actual_bet_size + previous_bet_this_street) - self.bet_to_match) * BB,
                                           (actual_bet_size + previous_bet_this_street) * BB)
                                          )
                self._change_bet_to_match(actual_bet_size + previous_bet_this_street)
                self.last_bet_placed_by = player
            else:
                raise Exception("Error when parsing action, make sure player action_type is PlayerAction and not int")

            should_transition_to_end = False
            players_with_actions = [p for p in self.players if p.state is PlayerState.ACTIVE if not p.all_in]
            players_who_should_act = [p for p in players_with_actions if (not p.acted_this_street or p.bet_this_street != self.bet_to_match)]

            # If the game is over, or the betting street is finished, progress the game state
            if len(players_with_actions) < 2 and len(players_who_should_act) == 0:
                amount = 0
                # If all active players are all-in, transition to the end, allowing no actions in the remaining streets
                if self.active_players > 1:
                    biggest_bet_call = max(
                        [p.bet_this_street for p in self.players
                         if p.state is PlayerState.ACTIVE if p is not self.last_bet_placed_by]
                    )
                    last_bet_this_street = 0
                    if self.last_bet_placed_by is not None:
                        last_bet_this_street = self.last_bet_placed_by.bet_this_street
                    if biggest_bet_call < last_bet_this_street:
                        amount = last_bet_this_street - biggest_bet_call
                    should_transition_to_end = True
                # If everyone else has folded, end the hand
                else:
                    self.hand_is_over = True
                    amount = self.minimum_raise
                # If there are uncalled bets, return them to the player who placed them
                if amount > 0:
                    self.pot -= amount
                    self.last_bet_placed_by.stack += amount
                    self.last_bet_placed_by.money_in_pot -= amount
                    self.last_bet_placed_by.bet_this_street -= amount
                    self._write_event(
                        "Uncalled bet ($%.2f) returned to %s" % (amount * BB, self.last_bet_placed_by.name)
                    )
                if should_transition_to_end:
                    self._street_transition(transition_to_end=True)
            # If the betting street is still active, choose next player to act
            else:
                active_players_after = [i for i in range(self.n_players) if i > self.current_player_i if
                                        self.players[i].state is PlayerState.ACTIVE if not self.players[i].all_in]
                active_players_before = [i for i in range(self.n_players) if i <= self.current_player_i if
                                         self.players[i].state is PlayerState.ACTIVE if not self.players[i].all_in]
                if len(active_players_after) > 0:
                    self.next_player_i = min(active_players_after)
                else:
                    self.next_player_i = min(active_players_before)
                next_player = self.players[self.next_player_i]
                if self.last_bet_placed_by is next_player or (self.first_to_act is next_player and self.last_bet_placed_by is None):
                    self.street_finished = True
                    if len(active_players_before) > 0:
                        self.next_player_i = min(active_players_before)

        if self.street_finished and not self.hand_is_over:
            self._street_transition()

        if self.hand_is_over:
            if self.final_rewards_collected == 0:
                self._distribute_pot()
                self._finish_hand()
            self.final_rewards_collected += 1
            active_players_after = [i for i in range(self.n_players) if i > self.current_player_i]
            active_players_before = [i for i in range(self.n_players) if i <= self.current_player_i]
            if len(active_players_after) > 0:
                self.next_player_i = min(active_players_after)
            else:
                self.next_player_i = min(active_players_before)

        return self._get_observation(self.players[self.next_player_i]), player.get_reward(), (self.hand_is_over and self.final_rewards_collected == self.n_players), {}

    def _street_transition(self, transition_to_end=False):
        transitioned = False
        if self.street == GameState.PREFLOP:
            self.cards = self.deck.draw(3)
            self._write_event("*** FLOP *** [%s %s %s]" %
                              (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                               Card.int_to_str(self.cards[2])))
            self.street = GameState.FLOP
            transitioned = True
        if self.street == GameState.FLOP and (not transitioned or transition_to_end):
            new = self.deck.draw(1)
            self.cards.append(new)
            self._write_event("*** TURN *** [%s %s %s] [%s]" %
                              (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                               Card.int_to_str(self.cards[2]), Card.int_to_str(self.cards[3])))
            self.street = GameState.TURN
            transitioned = True
        if self.street == GameState.TURN and (not transitioned or transition_to_end):
            new = self.deck.draw(1)
            self.cards.append(new)
            self._write_event("*** RIVER *** [%s %s %s %s] [%s]" %
                              (Card.int_to_str(self.cards[0]), Card.int_to_str(self.cards[1]),
                               Card.int_to_str(self.cards[2]), Card.int_to_str(self.cards[3]),
                               Card.int_to_str(self.cards[4])))
            self.street = GameState.RIVER
            transitioned = True
        if self.street == GameState.RIVER and (not transitioned or transition_to_end):
            if not self.hand_is_over:
                if self.hand_history_enabled:
                    self._write_show_down()
            self.hand_is_over = True
        self.street_finished = False
        self.last_bet_placed_by = None
        self.first_to_act = None
        self.bet_to_match = 0
        self.minimum_raise = 0
        for player in self.players:
            player.finish_street()

    def _change_bet_to_match(self, new_amount):
        self.minimum_raise = new_amount - self.bet_to_match
        self.bet_to_match = new_amount

    def _write_event(self, text):
        if self.hand_history_enabled:
            self.hand_history.append(text)

    def _history_initialize(self):
        t = time.localtime()
        self.hand_history.append("PokerStars Hand #%d: Hold'em No Limit ($%.2f/$%.2f USD) - %d/%d/%d %d:%d:%d ET" %
                            (np.random.randint(2230397, 32303976), SB, BB, t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour,
                             t.tm_min, t.tm_sec))
        self.hand_history.append("Table 'Wempe III' 6-max Seat #2 is the button")
        for i, player in enumerate(self.players):
            self.hand_history.append("Seat %d: %s ($%.2f in chips)" % (i+1, player.name, player.stack*BB))

    def _write_hole_cards(self):
        self.hand_history.append("*** HOLE CARDS ***")
        for i, player in enumerate(self.players):
            self.hand_history.append("Dealt to %s [%s %s]" %
                                (player.name, Card.int_to_str(player.cards[0]), Card.int_to_str(player.cards[1])))

    def _write_show_down(self):
        self.hand_history.append("*** SHOW DOWN ***")
        hand_types = [self.evaluator.class_to_string(self.evaluator.get_rank_class(p.hand_rank))
                        for p in self.players if p.state is PlayerState.ACTIVE]
        for player in self.players:
            if player.state is PlayerState.ACTIVE:
                player.calculate_hand_rank(self.evaluator, self.cards)
                player_hand_type = self.evaluator.class_to_string(self.evaluator.get_rank_class(player.hand_rank))
                matches = len([m for m in hand_types if m is player_hand_type])
                multiple = matches > 1
                self.hand_history.append("%s: shows [%s %s] (%s)" %
                                    (player.name, Card.int_to_str(player.cards[0]), Card.int_to_str(player.cards[1]),
                                     pretty_print_hand(player.cards, player_hand_type, self.cards, multiple))
                                    )

    def _finish_hand(self):
        for player in self.players:
            if self.hand_history_enabled:
                if player.winnings_for_hh > 0:
                    self._write_event("%s collected $%.2f from pot" % (player.name, player.winnings_for_hh*BB))

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
            with open('%s' % self.hand_history_location + 'handhistory_%s.txt' % time.time(), 'w') as f:
                for row in self.hand_history:
                    f.writelines(row + '\n')

    def _distribute_pot(self):
        pot = 0
        for player in self.players:
            if player.state is not PlayerState.ACTIVE:
                pot += player.money_in_pot
                player.winnings -= player.money_in_pot
        active_players = [p for p in self.players if p.state is PlayerState.ACTIVE]
        if len(active_players) == 1:
            active_players[0].winnings += pot
            active_players[0].winnings_for_hh += pot + active_players[0].money_in_pot
            return
        for player in active_players:
            player.calculate_hand_rank(self.evaluator, self.cards)
        while True:
            min_money_in_pot = min([p.money_in_pot for p in active_players])
            for player in active_players:
                pot += min_money_in_pot
                player.money_in_pot -= min_money_in_pot
                player.winnings -= min_money_in_pot
            best_hand_rank = min([p.hand_rank for p in active_players])
            winners = [p for p in active_players if p.hand_rank == best_hand_rank]
            for winner in winners:
                winner.winnings += pot / len(winners)
                winner.winnings_for_hh += pot / len(winners)
            active_players = [p for p in active_players if p.money_in_pot > 0]
            if len(active_players) <= 1:
                if len(active_players) == 1:
                    active_players[0].winnings += active_players[0].money_in_pot
                    active_players[0].winnings_for_hh += active_players[0].money_in_pot
                break
            pot = 0

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
            if not (approx_lte(bet_range[0], action.bet_amount) and approx_lte(action.bet_amount, bet_range[1])) or approx_gt(action.bet_amount, player.stack):
                if PlayerAction.FOLD in action_list:
                    player.fold()
                    self.active_players -= 1
                    self._write_event("%s: folds" % player.name)
                else:
                    player.check()
                    self._write_event("%s: checks" % player.name)
                return False
        return True

    def _get_valid_actions(self, player):
        valid_actions = [PlayerAction.CHECK, PlayerAction.FOLD, PlayerAction.BET, PlayerAction.CALL]
        valid_bet_range = [max(self.bet_to_match + self.minimum_raise, 1), player.stack]
        others_active = [p for p in self.players if p.state is PlayerState.ACTIVE if not p.all_in if p is not player]
        if self.bet_to_match == 0:
            valid_actions.remove(PlayerAction.CALL)
            valid_actions.remove(PlayerAction.FOLD)
        if self.bet_to_match != 0:
            valid_actions.remove(PlayerAction.CHECK)
        if player.stack < max(self.bet_to_match + self.minimum_raise, 1):
            valid_bet_range = [0, 0]
            valid_actions.remove(PlayerAction.BET)
        elif len(others_active) == 0:
            valid_actions.remove(PlayerAction.BET)
        return {'actions_list': valid_actions, 'bet_range': valid_bet_range}

    def _get_observation(self, player):
        observation = np.zeros(60, dtype=np.float32)
        observation[0] = player.identifier
        observation[1] = self.hand_is_over
        observation[2] = self.hand_ended_last_turn

        valid_actions = self._get_valid_actions(player)
        for action in valid_actions['actions_list']:
            observation[action.value+3] = 1
        observation[7] = valid_actions['bet_range'][0]
        observation[8] = valid_actions['bet_range'][1]

        observation[9] = player.position
        observation[10] = Card.get_suit_int(player.cards[0])
        observation[11] = Card.get_rank_int(player.cards[0])
        observation[12] = Card.get_suit_int(player.cards[1])
        observation[13] = Card.get_rank_int(player.cards[1])
        observation[14] = player.stack
        observation[15] = player.money_in_pot
        observation[16] = player.bet_this_street

        observation[17] = self.street
        for i in range(len(self.cards)):
            observation[18 + (i * 2)] = Card.get_suit_int(self.cards[i])
            observation[19 + (i * 2)] = Card.get_rank_int(self.cards[i])
        observation[22] = self.pot
        observation[23] = self.bet_to_match
        observation[24] = self.minimum_raise

        others = [other for other in self.players if other is not player]
        for i in range(len(others)):
            observation[25 + i * 6] = others[i].position
            observation[26 + i * 6] = others[i].state.value
            observation[27 + i * 6] = others[i].stack
            observation[28 + i * 6] = others[i].money_in_pot
            observation[29 + i * 6] = others[i].bet_this_street
            observation[30 + i * 6] = int(others[i].all_in)
        return observation