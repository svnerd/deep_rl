
# Multi-agent Actor-Critic algo for Tennis game

### The key points to the algo:

* This agent is implemented using multi-agent Actor-critic algo. The key idea is that during training time critic is
able to observe other agents behaviors and make the training more stable.

* During testing, actor only need the current observation state from itself to make decisions.  

* actor/critic has a target network to provide training targets and a local network to minimize the loss towards the target.

* The target network parameters are updated by local network by a blending coeff of 0.001 * local_network + 0.999 * target network every step.

* As shown in the detailed score chart, score.png, you can see that this multi-agent Actor/Critic structure is very stable as the avg score keeps rising.
![Image of scores](https://github.com/svnerd/deep_rl/blob/master/dpn/projects/tennis/report/score.png)


### raw outputs:
* record.csv


* successful message:
  *Environment solved in 1662 episodes!	Average Score: 1.005*
 


