
# DDPG agent for a robotic arm to reach an objects

### The key points to the algo:

*  This agent is implemented using DDPG algo with 2-layer fully connected NN for actor and critic.

* actor/critic has a target network to provide training targets and a local network to minimize the loss towards the target.

* The target network parameters are updated by local network by a blending coeff of 0.001 * local_network + 0.999 * target network every step.

* using A2C on multiple robotic arms at the same time to improve training stability. 

* As shown in the detailed score chart, score.png, you can see that this multi-agent Actor/Critic structure is very stable as the avg score keeps rising.
![Image of scores](https://github.com/svnerd/deep_rl/blob/master/dpn/projects/reacher/report/score.png)


### raw outputs:
* report/record.csv

* successful message:
  *Environment solved in 105 episodes!	Average Score: 30.061*

### future imporvements
* use lstm to see if it can capture time dependent nature of the robotic arm.
