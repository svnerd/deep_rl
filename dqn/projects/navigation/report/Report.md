
#  This DQN based agent implemented in Pytorch is able to navigate and collect bananas in a large, square world.


### The key points to the algo:

* This agent is implemented using Deep Q Network algo with a 2 fully connected layers of size [128, 64]

* The target network parameters are updated by local network by a blending coeff of 0.1 * local_network + 0.9 * target network every step.


### training results:
* As shown in the detailed score chart, score.png, you can see that this agent performance is quite stable as the avg score gets steady.
![Image of scores](https://github.com/svnerd/deep_rl/blob/master/dqn/projects/navigation/report/score.png)


### raw outputs:
* report/record.csv

* successful message:
  *Environment solved in 1472 episodes!	Average Score: 13.010*

### future improvements:
* use LSTM to capture time dependent nature of this task.
