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
The rewards are output as a numpy array, where the nth element corresponds to reward given to the agent, who was playing when the the observation acting player flag value was n.

**The acting player flag contained in the observation does not mean the agents position in the table**. 
Each player inside the table gets a unique id when the table instance is created, and this id is passed as the acting player flag in the observation.
This way agents can keep reacting to the same acting player flag value even after a table reset, while still playing from all possible table positions.

### Invalid actions
The environment deals with invalid actions by ignoring them, and either checking or folding automatically. 
If configured to do so, the environment also applies an invalid action penalty to the corresponding reward. The observation contains entries which can be used to implement invalid action masking.

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
        return table_action

    def reset(self):
        self.actions = []
        self.observations = []
        self.rewards = []

```


### Create an environment
```python
active_players = 6
agents = [ExampleRandomAgent() for _ in range(6)]
player_names = {0: 'TrackedAgent1', 1: 'Agent2'} # Rest are defaulted to player3, player4...
# Should we only log the 0th players (here TrackedAgent1) private cards to hand history files
track_single_player = True 
# Bounds for randomizing player stack sizes in reset()
low_stack_bbs = 50
high_stack_bbs = 200
hand_history_location = 'hands/'
invalid_action_penalty = 0
table = Table(active_players, 
              player_names=player_names,
              track_single_player=track_single_player,
              stack_low=low_stack_bbs,
              stack_high=high_stack_bbs,
              hand_history_location=hand_history_location,
              invalid_action_penalty=invalid_action_penalty
)
table.seed(1)
```

### Implement learning loop
```python
iteration = 1
while True:
    if iteration % 50 == 0:
        table.hand_history_enabled = True
    active_players = np.random.randint(2, 7)
    table.n_players = active_players
    obs = table.reset()
    for agent in agents:
        agent.reset()
    acting_player = int(obs[indices.ACTING_PLAYER])
    while True:
        action = agents[acting_player].get_action(obs)
        obs, reward, done, _ = table.step(action)
        if  done:
            # Distribute final rewards
            for i in range(active_players):
                agents[i].rewards.append(reward[i])
            break
        else:
            # This step can be skipped unless invalid action penalty is enabled, 
            # since we only get a reward when the pot is distributed, and the done flag is set
            agents[acting_player].rewards.append(reward[acting_player])
            acting_player = int(obs[indices.ACTING_PLAYER])
    iteration += 1
    table.hand_history_enabled = False
```
