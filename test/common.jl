using Test
using FastIce

using KernelAbstractions

# add KA backends
backends = KernelAbstractions.Backend[CPU(), ]

using CUDA
CUDA.functional() && push!(backends, CUDABackend())

using AMDGPU
AMDGPU.functional() && push!(backends, ROCBackend())
