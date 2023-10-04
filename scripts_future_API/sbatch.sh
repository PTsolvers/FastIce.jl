#!/bin/bash
#SBATCH --job-name=scaling_16
#SBATCH --account=project_465000557
#SBATCH --time=00:02:00
#SBATCH --nodes=4
#SBATCH --ntasks=16
#SBATCH --gpus-per-node=8
#SBATCH --partition=standard-g

# CPU_BIND="map_cpu:49,57,17,25,1,9,33,41"
CPU_BIND="map_cpu:49,17,1,33"

srun --cpu-bind=${CPU_BIND} ./profileme.sh