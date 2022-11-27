#!/bin/bash

# select platform
source ./setenv_lumi.sh

# srun -N1 -n8 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest ./runme.sh
# srun -N2 -n16 --ntasks-per-node=8 --gpus-per-node=8 --gpu-bind=closest ./runme.sh

# julia --project -e 'using AMDGPU; AMDGPU.versioninfo()'

# julia --project --check-bounds=no -O3 FastIce3D_visc_amd_opt_mpi.jl

julia --project --check-bounds=no -O3 FastIce3D_visc_amd_opt_mpi_perf.jl

# ./myrocprof --hsa-trace -d ./prof_out${SLURM_PROCID} -o ./prof_out${SLURM_PROCID}/results${SLURM_PROCID}.csv julia --project -O3 --check-bounds=no FastIce3D_visc_amd_opt_mpi_perf.jl
