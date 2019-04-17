  This DQN based agent implemented in Pytorch is able to navigate and collect bananas in a large, square world.

 You could clone the repo and in its root dir execute

 python -m deep_rl.dqn.projects.navigation.banana_controller --os [mac | linux]  --record-dir $your_dir


 This agent is implemented using Deep Q Network algo with a 2 fully connected layers of size [128, 64]


It contains a target network to provide training targets and a local network to minimize the loss towards the target.

The target network parameters are updated by local network by a blending coeff of 0.1 * local_network + 0.9 * target network every step.

Environment solved in 1472 episodes!	Average Score: 13.010
the detailed score chart is in score.png
the raw score csv is record.csv 
