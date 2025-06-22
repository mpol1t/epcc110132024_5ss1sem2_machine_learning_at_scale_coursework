#!/bin/bash

module load nvidia/cudnn/8.6.0-cuda-11.8
module load python/3.10.8-gpu
module load libsndfile/1.0.28
export PYTHONPATH=$PYTHONPATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-gpu/python/3.10.8/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-gpu/python/3.10.8/lib
export LIBRARY_PATH=$LIBRARY_PATH:/work/y07/shared/cirrus-software/pytorch/1.13.1-gpu/python/3.10.8/lib
export PYTHONUSERBASE=/work/m23oc/m23oc/$USER/python-installs
python3 -m pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
python -m pip install protobuf==3.20
python -m pip install --upgrade tensorboard
python3 -m pip install h5py
