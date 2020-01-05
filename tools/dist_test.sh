#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

GPUS=$1

CLUSTER=OFF $PYTHON -m torch.distributed.launch --nproc_per_node=$1 $(dirname "$0")/flow_test.py --launcher pytorch ${@:2}
