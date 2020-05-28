#!/usr/bin/env bash
export PATH="/mnt/lustre/share/cuda-9.0/bin:$PATH"
export LD_LIBRARY_PATH="/mnt/lustre/share/cuda-9.0/lib64:/mnt/lustre/share/nccl_2.1.15-1+cuda9.0_x86_64/lib:$LD_LIBRARY_PATH"

partition=$1
job_name=KntTest
img_list=$2
img_prefix=$3
checkpoint=$4
gpus=${5:-8}
gpu_per_node=${5:-8}

CLUSTER=ON srun -p ${partition} \
    --gres=gpu:${gpu_per_node} -n${gpus} --ntasks-per-node=${gpu_per_node} --cpus-per-task=2 \
    --job-name=${job_name} --kill-on-bad-exit=1 \
    python3 -u tools/vidfeat_extract.py --imglist=${img_list} --imgroot=${img_prefix} --checkpoint=${checkpoint} \
    --launcher='slurm' ${@:6} &
