using Test
using FastIce

using KernelAbstractions

function parse_flags!(args, flag; default=nothing, typ=typeof(default))
    for f in args
        startswith(f, flag) || continue

        if f != flag
            val = split(f, '=')[2]
            if !(typ ≡ nothing && typ <: AbstractString)
                val = parse(typ, val)
            end
        else
            val = default
        end

        filter!(x -> x != f, args)
        return true, val
    end
    return false, default
end

# add KA backends
backends = KernelAbstractions.Backend[CPU(), ]

_, backend_name = parse_flags!(ARGS, "--backend"; default="CPU", typ=String)

@static if backend_name == "AMDGPU"
    Pkg.add("AMDGPU")
    using AMDGPU
    AMDGPU.functional() && push!(backends, ROCBackend())
elseif backend_name == "CUDA"
    Pkg.add("CUDA")
    using CUDA
    CUDA.functional() && push!(backends, CUDABackend())
end