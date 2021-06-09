import numpy as np
import pokergym.obs_indices as indices
from pokergym.table import Table
from pokergym.common import PlayerAction, Action, action_list


class ExampleRandomAgent:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []

    def get_action(self, observation):
        # Only save the observation if it corresponds to an actual choice, not if the action to be taken is "don't care"
        if not observation[indices.ACTION_DONT_CARE]:
            self.observations.append(observation)
            valid_actions = np.argwhere(observation[indices.VALID_ACTIONS] == 1).flatten()
            valid_bet_low = observation[indices.VALID_BET_LOW]
            valid_bet_high = observation[indices.VALID_BET_HIGH]
            chosen_action = PlayerAction(np.random.choice(valid_actions))
            bet_size = 0
            if chosen_action is PlayerAction.BET:
                bet_size = np.random.uniform(valid_bet_low, valid_bet_high)
            table_action = Action(chosen_action, bet_size)
            self.actions.append(table_action)
        else:
            # If the action is "don't care", we are only feeding dummy actions to get the final end of hand rewards back
            table_action = Action(PlayerAction.CHECK, 0)
        return table_action

active_players = 6
agents = [ExampleRandomAgent() for _ in range(6)]
random_seed = 1
low_stack_bbs = 50
high_stack_bbs = 200
hh_location = 'hands/'
invalid_penalty = 0
table = Table(active_players, random_seed, low_stack_bbs, high_stack_bbs, hh_location, invalid_penalty)

### Implement learning loop
iteration = 1
while True:
    if iteration == 50:
        table.hand_history_enabled = True
        iteration = 0
    # Set a random number of players for each hand
    table.n_players = np.random.randint(2, 7)
    obs = table.reset()
    next_acting_player = int(obs[indices.ACTING_PLAYER])
    while True:
        action = agents[next_acting_player].get_action(obs)
        action_dont_care = obs[indices.ACTION_DONT_CARE]
        obs, reward, finished = table.step(action)

        if not action_dont_care:
            # If the action was not a "don't care", the reward corresponds to the action that we just took
            agents[next_acting_player].rewards.append(reward)
        else:
            # If the action was a "don't care", the reward is a delayed end of game reward which should correspond to the last valid action
            agents[next_acting_player].rewards[-1] += reward
        if finished:
            break
        next_acting_player = int(obs[indices.ACTING_PLAYER])
    iteration += 1
    table.hand_history_enabled = False