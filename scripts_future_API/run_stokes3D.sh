#!/bin/bash

module load LUMI/22.08
module load partition/G
module load rocm/5.3.3

export MPICH_GPU_SUPPORT_ENABLED=1

export LD_PRELOAD=${CRAY_MPICH_ROOTDIR}/gtl/lib/libmpi_gtl_hsa.so

julia --project -O3 tm_stokes_mpi_wip.jl
