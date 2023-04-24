#!/bin/bash

module purge
module load cuda openmpi hdf5

export JULIA_HDF5_PATH=/scratch-1/soft/spack/opt/spack/linux-ubuntu20.04-zen2/gcc-9.4.0/hdf5-1.12.1-cualmplov32dcc2bdnucqywutca437vp/

# julia --project scripts2D_variational_TM/test_volume_fractions2D.jl