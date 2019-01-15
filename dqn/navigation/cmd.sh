#!/bin/bash
DIR=$(pwd)/../../..
export PYTHONPATH=$DIR:$PYTHONPATH
set -ex
python -m drl.dqn.navigation.banana_controller
