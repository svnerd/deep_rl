#!/bin/bash
DIR=$(pwd)/../../..
export PYTHONPATH=$DIR:$PYTHONPATH
set -ex
python -m drl.dqn.gym.Deep_Q_Network
