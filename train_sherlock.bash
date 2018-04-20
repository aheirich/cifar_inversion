#!/bin/bash
#SBATCH --job-name=cifar-10
#SBATCH --time=08:00:00
#SBATCH -p aaiken
#SBATCH --gres gpu:1
#SBATCH --nodes=1

source ${HOME}/setup.bash
module load py-keras
module load py-tensorflow
cd ${HOME}/cifar_inversion

python train.py


