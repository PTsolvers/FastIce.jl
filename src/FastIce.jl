module FastIce

export field_array,get_device,get_backend,set_backend!
export to_device, to_host

using Preferences
using TinyKernels

const BACKEND = @load_preference("backend", "CPU")

@static if BACKEND == "CPU"
    const DEVICE = CPUDevice()
elseif BACKEND == "CUDA"
    using CUDA; @assert CUDA.functional()
    const DEVICE = CUDADevice()
elseif BACKEND == "AMDGPU"
    using AMDGPU; @assert AMDGPU.functional()
    const DEVICE = AMDGPUDevice()
elseif BACKEND == "Metal"
    using Metal; @assert Metal.functional()
    const DEVICE = MetalDevice()
else
    error("unsupported backend \"$BACKEND\"")
end

@inline get_device()  = DEVICE
@inline get_backend() = BACKEND

function set_backend!(new_backend)
    if !(new_backend ∈ ("CPU", "CUDA", "AMDGPU", "Metal"))
        throw(ArgumentError("invalid backend \"$new_backend\""))
    end
    @set_preferences!("backend" => new_backend)
    @info("new backend set; restart your Julia session for this change to take effect")
end

@inline field_array(::Type{T}, args...) where {T} = TinyKernels.device_array(T, DEVICE, args...)

@static if BACKEND == "CPU"
    @inline to_device(array::AbstractArray) = array
    @inline to_host(array::AbstractArray)   = array
elseif BACKEND == "CUDA"
    @inline to_device(array::CuArray)       = array
    @inline to_device(array::AbstractArray) = CuArray(array)
    @inline to_host(array::CuArray)         = Array(array)
    @inline to_host(array::AbstractArray)   = array
elseif BACKEND == "AMDGPU"
    @inline to_device(array::ROCArray)      = array
    @inline to_device(array::AbstractArray) = ROCArray(array)
    @inline to_host(array::ROCArray)        = Array(array)
    @inline to_host(array::AbstractArray)   = array
elseif BACKEND == "Metal"
    @inline to_device(array::MtlArray)      = array
    @inline to_device(array::AbstractArray) = MtlArray(array)
    @inline to_host(array::MtlArray)        = Array(array)
    @inline to_host(array::AbstractArray)   = array
end

end # module