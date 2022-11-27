#!/bin/bash

source setenv_lumi.sh

NZ=63  julia --project --check-bounds=no -O3 FastIce3D_visc_amd_opt_perf.jl
NZ=127 julia --project --check-bounds=no -O3 FastIce3D_visc_amd_opt_perf.jl
NZ=255 julia --project --check-bounds=no -O3 FastIce3D_visc_amd_opt_perf.jl
NZ=383 julia --project --check-bounds=no -O3 FastIce3D_visc_amd_opt_perf.jl
NZ=511 julia --project --check-bounds=no -O3 FastIce3D_visc_amd_opt_perf.jl
NZ=575 julia --project --check-bounds=no -O3 FastIce3D_visc_amd_opt_perf.jl
