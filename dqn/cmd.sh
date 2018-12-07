#!/bin/bash
DIR=$(pwd)/../..
export PYTHONPATH=$DIR:$PYTHONPATH
set -ex
python -m rl.dqn.Deep_Q_Network
