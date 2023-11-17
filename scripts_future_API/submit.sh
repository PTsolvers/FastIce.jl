#!/bin/bash -l
#SBATCH --job-name="FastIce3D"
#SBATCH --output=FastIce3D.%j.o
#SBATCH --error=FastIce3D.%j.e
#SBATCH --time=00:05:00
#SBATCH --nodes=16
#SBATCH --ntasks=64
# #SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=standard-g
#SBATCH --account project_465000557

# export ROCR_VISIBLE_DEVICES=0,2,4,6

srun --cpu-bind=map_cpu:49,17,1,33 ./profileme.sh
