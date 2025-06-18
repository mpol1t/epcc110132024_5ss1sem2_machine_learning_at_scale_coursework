#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --hint=multithread
#SBATCH --gres=gpu:1
#SBATCH --account=m23oc-s1548787

module load nvidia/cudnn/8.6.0-cuda-11.8
module load python/3.10.8-gpu
module load libsndfile/1.0.28

export PYTHONPATH=$PYTHONPATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-gpu/python/3.10.8/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-gpu/python/3.10.8/lib
export LIBRARY_PATH=$LIBRARY_PATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-gpu/python/3.10.8/lib

export MPLCONFIGDIR=/work/m23oc/m23oc/u1548787/mplconfig/

export PYTHONUSERBASE=/work/m23oc/m23oc/$USER/python-installs
export PYTHONPATH=/work/m23oc/m23oc/$USER/python-installs/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/work/m23oc/m23oc/$USER/python-installs/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

srun -n 1 -c 10 python3 train.py --config short

