module FastIce

export field_array,get_device,get_backend,set_backend!
export to_device, to_host

using Preferences
using TinyKernels

const BACKEND = @load_preference("backend","CUDA")

@static if BACKEND == "CUDA"
    using CUDA
    using TinyKernels.CUDABackend
    const DEVICE = CUDABackend.CUDADevice()
elseif BACKEND == "AMDGPU"
    using AMDGPU
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
@static if BACKEND == "CUDA"
    @inline to_device(array::CuArray)       = array
    @inline to_device(array::AbstractArray) = CuArray(array)
    @inline to_host(array::CuArray)         = Array(array)
    @inline to_host(array::AbstractArray)   = array
elseif BACKEND == "AMDGPU"
    @inline to_device(array::ROCArray)      = array
    @inline to_device(array::AbstractArray) = ROCArray(array)
    @inline to_host(array::ROCArray)        = Array(array)
    @inline to_host(array::AbstractArray)   = array
end


# include("level_sets/level_sets.jl")
# include("geometry.jl")

end # module
