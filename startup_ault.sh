#!/bin/bash

source ./scripts3D_amdgpu/setenv_ault.sh

julian --project -e 'using Pkg; Pkg.resolve()'

julian --project -e 'using Pkg; pkg"add https://github.com/luraess/ImplicitGlobalGrid.jl#lr/amdgpu-0.4.x-support";'

julian --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'

julian --project -e 'using AMDGPU; AMDGPU.versioninfo()'
