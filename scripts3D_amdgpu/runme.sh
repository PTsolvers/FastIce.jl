#!/bin/bash

# source start.sh
# mpirun -np 1 --bind-to socket julia --project -O3 --check-bounds=no SteadyStateGlacier3D_indices_xpu.jl
# mpirun -np 4 --bind-to socket julia --project -O3 --check-bounds=no SteadyStateGlacier3D_indices_xpu.jl
# mpirun -np 8 --bind-to socket julia --project -O3 --check-bounds=no SteadyStateGlacier3D_TM_xpu.jl

# select platform
source ./setenv_ault.sh
# source ./setenv_lumi.sh

# ~/julia_local/julia-1.9-dev/bin/julia --project -O3 --check-bounds=no SteadyStateGlacier3D_TM_indices_xpu.jl
~/julia_local/julia-1.9-dev/bin/julia --project SteadyStateGlacier3D_TM_indices_xpu.jl
