#!/bin/bash
#SBATCH --job-name=cifarCPU
#SBATCH --time=32:00:00
#SBATCH -p aaiken
#SBATCH --nodes=1

source ${HOME}/setup.bash
module load py-keras
module load py-tensorflow
cd ${HOME}/cifar_inversion

python train.py


