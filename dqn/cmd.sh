#!/bin/bash
DIR=$(pwd)/../
export PYTHONPATH=$DIR:$PYTHONPATH
set -ex
python -m dqn.dqn_agent 
