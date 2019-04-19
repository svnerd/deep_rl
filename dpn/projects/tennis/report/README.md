  
# This DQN based agent implemented in Pytorch enables two agents to play tennis. 

### install and execute instruction:

* install prerequisits with
`pip install pytorch`


* Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Linux (headless): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

* then clone codebase:

`git clone git@github.com:svnerd/deep_rl.git`


### to train from scratch:

 `python -m deep_rl.dpn.projects.tennis.tennis --os [mac | linux]  --record-dir $your_dir`

### to load the trained models:
  `python -m deep_rl.dpn.projects.tennis.tennis --os [mac | linux]  --record-dir ${your_dir}/saved_models`
