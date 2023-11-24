#!/bin/bash -l
#SBATCH --job-name="FastIce3D"
#SBATCH --output=FastIce3D.%j.o
#SBATCH --error=FastIce3D.%j.e
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks=16
# #SBATCH --ntasks-per-node=8 # this somehow fails...
#SBATCH --gpus-per-node=8
#SBATCH --partition=standard-g
#SBATCH --account project_465000557

# CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"

# export ROCR_VISIBLE_DEVICES=0,2,4,6 # -> done in runme.sh
CPU_BIND="map_cpu:49,17,1,33"

srun --cpu-bind=${CPU_BIND} ./runme.sh
