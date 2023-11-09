module AMDGPUExt

using AMDGPU, AMDGPU.ROCKernels

import FastIce.Architecture: heuristic_groupsize, set_device!

set_device!(dev::HIPDevice) = AMDGPU.device!(dev)

get_device(::ROCBackend, id::Integer) = HIPDevice(id)

heuristic_groupsize(::HIPDevice, ::Val{1}) = (256, )
heuristic_groupsize(::HIPDevice, ::Val{2}) = (128, 2, )
heuristic_groupsize(::HIPDevice, ::Val{3}) = (128, 2, 1, )

end
