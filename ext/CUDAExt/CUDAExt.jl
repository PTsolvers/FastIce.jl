module CUDAExt

using CUDA
using KernelAbstractions

using FastIce.Architecture

set_device!(dev::CuDevice) = device!(dev)

heuristic_groupsize(::CUDABackend, ::Val{1}) = (256, )
heuristic_groupsize(::CUDABackend, ::Val{2}) = (32, 8, )
heuristic_groupsize(::CUDABackend, ::Val{3}) = (32, 8, 1, )

end