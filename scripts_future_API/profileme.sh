#!/bin/bash

source /users/lurass/scratch/setenv_lumi.sh

# previous
# srun -N1 -n2 --ntasks-per-node=2 --gpus-per-node=2 --gpu-bind=closest profileme.sh

# basic
# srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 -N1 -n8 --gpus-per-node=8 profileme.sh

# optimal using only singelGCD per Module
# srun --cpu-bind=map_cpu:49,17,1,33 -N1 -n1 --gpus-per-node=8 profileme.sh
# srun --cpu-bind=map_cpu:49,17,1,33 -N4 -n16 --gpus-per-node=8 profileme.sh
export ROCR_VISIBLE_DEVICES=0,2,4,6

# julia --project bench3d.jl
# julia --project exchanger2.jl
# julia --project rocmaware.jl
julia --project benchmark_diffusion_3D.jl

# ENABLE_JITPROFILING=1 ../../myrocprof --hsa-trace --hip-trace julia --project ./exchanger2.jl

# ENABLE_JITPROFILING=1 rocprof --hip-trace --hsa-trace -d ./prof_out${SLURM_PROCID} -o ./prof_out${SLURM_PROCID}/results${SLURM_PROCID}.csv julia --project bench3d.jl
