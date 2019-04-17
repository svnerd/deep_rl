  This DQN based agent implemented in Pytorch enables a double-jointed arm to move to target locations.

 You could clone the repo and in its root dir execute

 python -m deep_rl.dpn.projects.reacher.reacher --os [mac | linux]  --record-dir $your_dir


 This agent is implemented using DDPG algo with 2-layer fully connected NN for actor and critic. 


actor/critic has a target network to provide training targets and a local network to minimize the loss towards the target.

The target network parameters are updated by local network by a blending coeff of 0.001 * local_network + 0.999 * target network every step.

Environment solved in 105 episodes!	Average Score: 30.061

the detailed score chart is in score.png

the raw score csv is record.csv 
