#!/bin/bash

module load LUMI/22.08
module load partition/G
module load rocm/5.3.3

# ROCm-aware MPI set to 1, else 0
export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_ROCMAWARE_MPI=1

# Needs to know about location of GTL lib
export LD_PRELOAD=${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so

julia --project rocmaware.jl