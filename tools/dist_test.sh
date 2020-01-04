#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

CLUSTER=OFF $PYTHON -m torch.distributed.launch --nproc_per_node=$3 $(dirname "$0")/test.py $1 $2 --launcher pytorch ${@:4}
