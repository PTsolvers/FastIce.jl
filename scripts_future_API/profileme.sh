#!/bin/bash

source /users/lurass/scratch/setenv_lumi.sh

# srun -N1 -n2 --ntasks-per-node=2 --gpus-per-node=2 --gpu-bind=closest profileme.sh

# srun --cpu-bind=map_cpu:49,57,17,25,1,9,33,41 -N1 -n8 --gpus-per-node=8 profileme.sh

# julia --project laplacian_3d.jl
julia --project bench3d.jl
# julia --project exchanger2.jl
# julia --project rocmaware.jl

# ENABLE_JITPROFILING=1 ../../myrocprof --hsa-trace --hip-trace julia --project ./exchanger2.jl

# export HIP_VISIBLE_DEVICES=$SLURM_LOCALID
# export HIP_VISIBLE_DEVICES="0,2,4,6"

# ENABLE_JITPROFILING=1 rocprof --hsa-trace --hip-trace -d ./prof_out${SLURM_PROCID} -o ./prof_out${SLURM_PROCID}/results${SLURM_PROCID}.csv julia --project bench3d.jl