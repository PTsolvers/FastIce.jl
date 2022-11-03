#!/bin/bash

# module load hip-rocclr/4.2.0 hip/4.2.0 hsa-rocr-dev/4.2.0 hsakmt-roct/4.2.0 llvm-amdgpu/4.2.0 rocm-cmake/4.2.0 rocminfo/4.2.0 roctracer-dev-api/4.2.0
module load rocm hip-rocclr hip hsa-rocr-dev hsakmt-roct llvm-amdgpu rocm-cmake rocminfo roctracer-dev-api rocprofiler-dev rocm-smi-lib

export JULIA_AMDGPU_DISABLE_ARTIFACTS=1

export SLURM_MPI_TYPE=pmix
export UCX_WARN_UNUSED_ENV_VARS=n

# ROCm-aware MPI
module load roc-ompi
export IGG_ROCMAWARE_MPI=1

# Standard MPI
# export PMIX_MCA_psec=native
# module load openmpi
# export IGG_ROCMAWARE_MPI=0

echo "ENV setup done"
