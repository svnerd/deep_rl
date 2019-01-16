#!/bin/bash
DIR=$(pwd)/../../..
export PYTHONPATH=$DIR:$PYTHONPATH
set -ex
python -m drl.dpn.cartpole.pole_controller
