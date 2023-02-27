#!/bin/bash

module load CrayEnv
module load craype-accel-amd-gfx90a # MI250x
module load cray-mpich
module load rocm

export JULIA_AMDGPU_DISABLE_ARTIFACTS=1

# ROCm-aware MPI set to 1, else 0
export MPICH_GPU_SUPPORT_ENABLED=1
export IGG_ROCMAWARE_MPI=1

# Needs to know about location of GTL lib
export LD_PRELOAD=${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so

echo "ENV setup done"
