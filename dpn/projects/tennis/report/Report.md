
# Multi-agent Actor-Critic algo for Tennis game


### Learning Algorithm
* This agent is implemented using multi-agent Actor-critic algo. The key idea is that during training time critic is
able to observe other agents behaviors and make the training more stable.
* During testing, actor only need the current observation state from itself to make decisions.

### Model architectures
* actor/critic has a target network to provide training targets and a local network to minimize the loss towards the target.
* Both network are 2-layer [256, 128] fully connected NN.
* inputs of critic contains observations and all agents actions.

### Hyperparameters
* DISCOUNT_RATE = 0.99
* TAU = 1e-3      for soft update of target parameters
* LR_ACTOR = 4e-4         learning rate of the actor
* LR_CRITIC = 4e-4        learning rate of the critic

Plot of Rewards

* As shown in the detailed score chart, score.png, you can see that this multi-agent Actor/Critic structure is very stable as the avg score keeps rising.
![Image of scores](https://github.com/svnerd/deep_rl/blob/master/dpn/projects/tennis/report/score.png)

### raw outputs:
* record.csv
* successful message:
  *Environment solved in 1662 episodes!	Average Score: 1.005*
 
### future imporvement:
* try modelling from pixel with CNN so that critic and actor have the global view of the states.

