#!/bin/bash

source /users/lurass/scratch/setenv_lumi.sh
# module load LUMI/23.09
# module load partition/G
# module load rocm/5.6.1

# ROCm-aware MPI set to 1, else 0
export MPICH_GPU_SUPPORT_ENABLED=1

## basic
# srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 -N1 -n8 --gpus-per-node=8 runme.sh

## optimal using only single GCD per MI250x Module
# srun --cpu-bind=map_cpu:49,17,1,33 -N1 -n1 --gpus-per-node=8 runme.sh
# srun --cpu-bind=map_cpu:49,17,1,33 -N4 -n16 --gpus-per-node=8 runme.sh
export ROCR_VISIBLE_DEVICES=0,2,4,6

# julia --project benchmark_diffusion_3D.jl
julia --project --color=yes tm_stokes_mpi_wip.jl

# Profiling
# ENABLE_JITPROFILING=1 rocprof --hip-trace --hsa-trace -d ./prof_out${SLURM_PROCID} -o ./prof_out${SLURM_PROCID}/results${SLURM_PROCID}.csv julia --project bench3d.jl
