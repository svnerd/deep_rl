# This DQN based agent implemented in Pytorch is able to navigate and collect bananas in a large, square world.
.

### install and execute instruction:
`git clone git@github.com:svnerd/deep_rl.git`

### to train from scratch:

 `python -m deep_rl.dqn.projects.navigation.banana_controller --os [mac | linux]  --record-dir $your_dir`

### to load the trained models:
  `python -m deep_rl.dqn.projects.navigation.banana_controller --os [mac | linux]  --record-dir ${your_dir}/saved_models`
