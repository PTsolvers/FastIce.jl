#!/bin/bash

source start.sh

mpirun -np 1 --bind-to socket julia --project -O3 --check-bounds=no SteadyStateGlacier3D_indices_xpu.jl

# mpirun -np 4 --bind-to socket julia --project -O3 --check-bounds=no SteadyStateGlacier3D_indices_xpu.jl
# mpirun -np 8 --bind-to socket julia --project -O3 --check-bounds=no SteadyStateGlacier3D_TM_xpu.jl
