#!/bin/bash

#SBATCH -J supcon_baseline
#SBATCH -o supcon_baseline%j.out
#SBATCH -e supcon_baseline%j.err
#SBATCH --mail-user=swhan@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=1T
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG
echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date

./distributed_train.sh 4 /nobackup/projects/public/imagenet21k_resized/ \
--train-split imagenet21k_train --val-split imagenet21k_val \
-b 2048 --model resnet50 --num-classes 0 --epochs 300 --opt lamb --sched cosine --lr 0.005 --amp \
--weight-decay 0.02 --warmup-epochs 5 --eval-metric loss \
--hflip 0.5 --aa rand-m7-mstd0.5 --mixup 0.1 --cutmix 1.0 --drop-path 0.05 \
--aug-splits 4 --dist-bn reduce 

echo "Run completed at:- "
date

#!/bin/bash