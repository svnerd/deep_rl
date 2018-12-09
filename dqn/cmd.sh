#!/bin/bash
DIR=$(pwd)/../..
export PYTHONPATH=$DIR:$PYTHONPATH
set -ex
python -m drl.dqn.Deep_Q_Network
