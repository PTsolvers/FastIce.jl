#!/bin/bash

source /users/lurass/scratch/setenv_lumi.sh

# basic
# srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 -N1 -n8 --gpus-per-node=8 profileme.sh

# optimal using only single GCD per MI250x Module
# srun --cpu-bind=map_cpu:49,17,1,33 -N1 -n1 --gpus-per-node=8 profileme.sh
# srun --cpu-bind=map_cpu:49,17,1,33 -N4 -n16 --gpus-per-node=8 profileme.sh
export ROCR_VISIBLE_DEVICES=0,2,4,6

# julia --project benchmark_diffusion_3D.jl
julia --project --color=yes tm_stokes_mpi_wip.jl

# ENABLE_JITPROFILING=1 rocprof --hip-trace --hsa-trace -d ./prof_out${SLURM_PROCID} -o ./prof_out${SLURM_PROCID}/results${SLURM_PROCID}.csv julia --project bench3d.jl
