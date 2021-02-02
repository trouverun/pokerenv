# Pokergym
Pokergym is a reinforcement learning environment for No Limit Texas Hold'em. 

In order to make the turn based multiagent system more natural, the environment operates in the opposite way to the OpenAI gym interface. 
Here the environment calls step() on an agent to get and action given observation, rather than the other way around.
This way neither the main training loop nor the agents have to have any understanding of the game logic, e.g. whose turn it is, since that is handled internally by the environment.

The results of a hand can be printed in to a hand history file, which can be analyzed with any pokerstars compatible tracking software, allowing you to easily track the learning process.

## Example

### Define an agent

```python
ExampleRandomAgent(BaseAgent):
    def __init__(self, identifier):
        self.identifier = identifier
        self.actions = []
        self.observations = []
        self.rewards = []
    
    def step(self, observation, valid_actions, previous_reward, episode_over):
        if previous_reward is not None:
            self.rewards.append(reward)
        if episode over:
            return
        self.observations.append(observation)
        actions_list, bet_range = valid_actions['actions_list'], valid_actions['bet_range']
        action = PlayerAction(random.choice(actions_list))
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
agents = [ExampleAgent('example_agent_%d' % i) for i in range(6)]
random_seed = 1
low_stack_bbs = 50
high_stack_bbs = 200
hh_location = 'hands'/
invalid_penalty = 0

table = Table(active_play, agents, random_seed, low_stack_bbs, high_stack_bbs, hh_location, invalid_penalty)
```

### Implement learning loop
```python
while True:
  table.reset()
  table.play_hand()
```
