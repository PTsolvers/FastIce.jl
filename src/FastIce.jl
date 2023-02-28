module FastIce

export field_array,get_device,get_backend,set_backend!

using Preferences
using TinyKernels

const BACKEND = @load_preference("backend","CUDA")

@static if BACKEND == "CUDA"
    using CUDA
    @assert CUDA.functional()
    using TinyKernels.CUDABackend
    const DEVICE = CUDABackend.CUDADevice()
elseif BACKEND == "AMDGPU"
    using AMDGPU
    @assert AMDGPU.functional()
    using TinyKernels.ROCBackend
    const DEVICE = ROCBackend.ROCDevice()
else
    error("unsupported backend \"$BACKEND\"")
end

@inline get_device()  = DEVICE
@inline get_backend() = BACKEND

function set_backend!(new_backend)
    if !(new_backend ∈ ("CUDA", "AMDGPU"))
        throw(ArgumentError("invalid backend \"$new_backend\""))
    end
    @set_preferences!("backend" => new_backend)
    @info("new backend set; restart your Julia session for this change to take effect")
end

@inline field_array(::Type{T}, args...) where T = TinyKernels.device_array(T,DEVICE,args...)

include("level_sets/level_sets.jl")
include("geometry.jl")

end # module
