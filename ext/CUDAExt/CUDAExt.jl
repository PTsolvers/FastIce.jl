module CUDAExt

using CUDA

import FastIce.Architecture: heuristic_groupsize, set_device!

set_device!(dev::CuDevice) = CUDA.device!(dev)

set_device!(::CuDevice, id::Integer) = CUDA.device!(id-1)

heuristic_groupsize(::CuDevice, ::Val{1}) = (256, )
heuristic_groupsize(::CuDevice, ::Val{2}) = (32, 8, )
heuristic_groupsize(::CuDevice, ::Val{3}) = (32, 8, 1, )

end
