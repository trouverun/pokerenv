# Pokerenv
Pokerenv is an openAI gym (https://gym.openai.com/docs/) compliant reinforcement learning environment for No Limit Texas Hold'em. It supports 2-6 player tables.

The environment can be configured to output hand history files, which can be viewed with any pokerstars compatible tracking software (holdem manager, pokertracker, etc.), allowing you to easily track the learning process.

## Installation and dependencies
```shell
pip install numpy
pip install treys
pip install pokerenv
```

## Usage information 
### Working around delayed rewards
Due to the fact that final rewards in NL hold'em are sometimes dependent on the actions other players take after you, some extra logic in the learning loop is required to assign rewards to actions taken, which means that <em>the environment can't be directly used with any baseline RL algorithms</em>, which assume that all rewards correspond the action just taken. 

In the learning loop you will have to watch out for two flags in the observation. The HAND_IS_OVER flag signals that the environment is in reward collection mode, where it will ask all players for dummy actions to distribute final rewards. When the HAND_IS_OVER flag is set, all actions passed in to the step() function are ignored, making them "don't cares". The other DELAYED_REWARD flag signals that the reward which was just output by the environment was a result of a "don't care" action, and that the reward should be added to the last valid action taken (by the previously acting player).

### Invalid actions
The environment deals with invalid actions by ignoring them, and either checking or folding automatically. If configured to do so, the environment also applies an invalid action penalty to the corresponding reward. The observation contains entries which can be used to implement invalid action masking.

All of the required (from the learning loop perspective) observation entries have human readable index definitions in the obs_indices.py module.

## Toy example

### Define an agent

```python
import numpy as np
import pokerenv.obs_indices as indices
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action, action_list


class ExampleRandomAgent:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []

    def get_action(self, observation):
        # If the hand is over, the environment is asking for dummy actions to distribute final rewards.
        # This means that the action is a don't care, and will be ignored by the environment.
        # This also means, that the observation does not correspond to any meaningful choice to be taken, 
        # and it should be ignored as well.
        if not observation[indices.HAND_IS_OVER]:
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
            table_action = Action(PlayerAction.CHECK, 0)
        return table_action
```


### Create an environment
```python
active_players = 6
agents = [ExampleRandomAgent() for _ in range(6)]
# Bounds for randomizing player stack sizes in reset()
low_stack_bbs = 50
high_stack_bbs = 200
hand_history_location = 'hands/'
invalid_action_penalty = 0
table = Table(active_players, low_stack_bbs, high_stack_bbs, hand_history_location, invalid_penalty)
table.seed(1)
```

### Implement learning loop
```python
iteration = 1
while True:
    if iteration == 50:
        table.hand_history_enabled = True
        iteration = 0
    table.n_players = np.random.randint(2, 7)
    obs = table.reset()
    acting_player = int(obs[indices.ACTING_PLAYER])
    while True:
        action = agents[acting_player].get_action(obs)
        obs, reward, done, _ = table.step(action)
        # If the reward is delayed, we are collecting end of game rewards by feeding in dummy actions
        delayed_reward = obs[indices.DELAYED_REWARD]
        
        if  delayed_reward:
            # If the reward is delayed, the action we just took was a don't care, 
            # and the reward corresponds to the last valid action taken
            agents[acting_player].rewards[-1] += reward
        else:
            # Otherwise the reward corresponds to the action we just took
            agents[acting_player].rewards.append(reward)
        if done:
            break
        acting_player = int(obs[indices.ACTING_PLAYER])
    iteration += 1
    table.hand_history_enabled = False
```
