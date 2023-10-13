module AMDGPUExt

using AMDGPU
using KernelAbstractions

using FastIce.Architecture

set_device!(dev::HIPDevice) = AMDGPU.device!(dev)

set_device!(::ROCBackend, id::Integer) = AMDGPU.device_id!(id)

heuristic_groupsize(::ROCBackend, ::Val{1}) = (256, )
heuristic_groupsize(::ROCBackend, ::Val{2}) = (128, 2, )
heuristic_groupsize(::ROCBackend, ::Val{3}) = (128, 2, 1, )

end