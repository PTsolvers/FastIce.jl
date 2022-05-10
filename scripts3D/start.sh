#!/bin/bash

module load julia openmpi cuda hdf5

export JULIA_HDF5_PATH=$HDF5_ROOT
export JULIA_CUDA_MEMORY_POOL=none
export IGG_CUDAAWARE_MPI=1
echo ready
