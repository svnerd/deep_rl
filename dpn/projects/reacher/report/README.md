# This DQN based agent implemented in Pytorch enables a double-jointed arm to move to target locations.

### install and execute instruction:
`git clone git@github.com:svnerd/deep_rl.git`

### to train from scratch:

  `python -m deep_rl.dpn.projects.reacher.reacher --os [mac | linux]  --record-dir $your_dir`

### to load the trained models:
  `python -m deep_rl.dpn.projects.reacher.reacher --os [mac | linux]  --record-dir ${your_dir}/saved_models`
