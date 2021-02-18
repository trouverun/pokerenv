# Pokerenv
Pokerenv is an openAI gym (https://gym.openai.com/docs/) compliant reinforcement learning environment for No Limit Texas Hold'em. It supports 2-6 player tables.

The environment can be configured to output hand history files, which can be viewed with any pokerstars compatible tracking software (holdem manager, pokertracker, etc.), allowing you to easily track the learning process.

The observation space and other details are described in the wiki (WIP): https://github.com/trouverun/pokerenv/wiki/ 

## Installation
```shell
pip install pokerenv
```

## Toy example

### Define an agent

```python
import numpy as np
from pokergym.table import Table
from pokergym.common import PlayerAction, Action


class ExampleRandomAgent:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []

    def get_action(self, observation):
        # Only save valid observations
        if not observation['info']['hand_is_over']:
            self.observations.append(observation)
            action_list = observation['info']['valid_actions']['actions_list']
            bet_range = observation['info']['valid_actions']['bet_range']
            chosen = PlayerAction(np.random.choice(action_list))
            betsize = 0
            if chosen is PlayerAction.BET:
                betsize = np.random.uniform(bet_range[0], bet_range[1])
            action = Action(chosen, betsize)
            self.actions.append(action)
        else:
            # Hand is over and we are only collecting final rewards, actions are ignored,
            # so send a dummy action without recording it
            action = Action(0, 0)
        return action

```


### Create an environment
```python
active_players = 6
agents = [ExampleRandomAgent() for _ in range(6)]
random_seed = 1
low_stack_bbs = 50
high_stack_bbs = 200
hh_location = 'hands/'
invalid_penalty = -15

table = Table(active_players, random_seed, low_stack_bbs, high_stack_bbs, hh_location, invalid_penalty, obs_format='dict')
```

### Implement learning loop
```python
iteration = 1
while True:
    if iteration == 50:
        table.hand_history_enabled = True
        iteration = 0
    # Set a random number of players each hand
    table.n_players = np.random.randint(2, 7)
    obs = table.reset()
    next_acting_player = obs['info']['next_player_to_act']
    while True:
        action = agents[next_acting_player].get_action(obs)
        obs, reward, finished = table.step(action)
        
        # Check if the reward corresponds to the previous action taken, 
        # or if it is a delayed reward given at the end of a game (should be added to latest reward)
        if not obs['info']['delayed_reward']:
            agents[next_acting_player].rewards.append(reward)
        else:
            agents[next_acting_player].rewards[-1] += reward
        
        if finished:
            break
        next_acting_player = obs['info']['next_player_to_act']
    iteration += 1
    
    table.hand_history_enabled = False
  
```
