# Pokerenv
Pokerenv is a reinforcement learning environment for No Limit Texas Hold'em. 

In order to make the turn based multiagent system more natural, the environment operates in the opposite way when compared to the OpenAI gym interface. 
Here the environment calls step() on an agent to get and action given observation, rather than the other way around.
This way neither the main training loop nor the agents need to have any understanding of the game logic, e.g. whose turn it is, since that is handled internally by the environment.

The results of a hand can be printed in to a hand history file, which can be analyzed with any pokerstars compatible tracking software, allowing you to easily track the learning process.

The intended use case is to run massive amounts of actor processes interacting with the environment, querying a separate learner for actions, and updating the policies based on trajectories formed inside the learner, as per the SEED architecture. (https://ai.googleblog.com/2020/03/massively-scaling-reinforcement.html).

## Installation
### Requirements
```shell
pip install numpy
pip install treys
pip install pokerenv
```

## Example

### Define an agent

```python
from pokerenv.table import Table
from pokerenv.common import PlayerAction, Action, BaseAgent

class ExampleRandomAgent(BaseAgent):
    def __init__(self, identifier):
        self.actions = []
        self.observations = []
        self.rewards = []
    
    def step(self, observation, valid_actions, previous_reward, episode_over):
        if previous_reward is not None:
            self.rewards.append(previous_reward)
        if episode_over:
            return
        self.observations.append(observation)
        actions_list, bet_range = valid_actions['actions_list'], valid_actions['bet_range']
        chosen = PlayerAction(random.choice(actions_list))
        betsize = 0
        if chosen is PlayerAction.BET:
            betsize = random.uniform(bet_range[0], bet_range[1])
        action = Action(chosen, betsize)
        self.actions.append(action)
        return action
```


### Create an environment
```python
active_players = 6
agents = [ExampleRandomAgent() for i in range(6)]
random_seed = 1
low_stack_bbs = 50
high_stack_bbs = 200
hh_location = 'hands/'
invalid_penalty = 0

table = Table(active_players, agents, random_seed, low_stack_bbs, high_stack_bbs, hh_location, invalid_penalty)
```

### Implement learning loop
```python
iteration = 1
while True:
  if iteration == 50:
      table.hand_history_enabled = True
      iteration = 0
  table.reset()
  table.play_hand()
  table.hand_history_enabled = False
  iteration += 1
  
```
