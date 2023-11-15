#!/bin/bash -l
#SBATCH --job-name="FastIce3D"
#SBATCH --output=FastIce3D.%j.o
#SBATCH --error=FastIce3D.%j.e
#SBATCH --time=00:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=standard-g
#SBATCH --account project_465000557

srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 ./run_stokes3D.sh
