#!/bin/bash

# -> to make it executable: chmod +x runme.sh or chmod 755 runme.sh

nsys profile --cpuctxsw=none --sample=none --trace=cuda --output=outprof.%q{OMPI_COMM_WORLD_RANK} -f true julia --project -O3 --check-bounds=no SteadyStateGlacier3D_xpu.jl
