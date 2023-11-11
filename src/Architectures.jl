module Architectures

export Architecture

export launch!, set_device!, get_device, set_device_and_priority!, heuristic_groupsize
export backend, device, details

using FastIce.Grids

using KernelAbstractions

struct Architecture{Kind,B,D,Details}
    backend::B
    device::D
    details::Details
end

struct SingleDevice end

function Architecture(backend::Backend, device_id::Integer=1)
    device = get_device(backend, device_id)
    return Architecture{SingleDevice,typeof(backend),typeof(device),Nothing}(backend, device, nothing)
end

device(arch::Architecture) = arch.device
backend(arch::Architecture) = arch.backend
details(arch::Architecture) = arch.details

set_device!(arch::Architecture) = set_device!(arch.device)

function set_device_and_priority!(arch::Architecture, prio::Symbol)
    set_device!(arch)
    KernelAbstractions.priority!(arch.backend, prio)
    return
end

set_device!(::Architecture{Kind,CPU}) where {Kind} = nothing
get_device(::CPU, id::Integer) = nothing

heuristic_groupsize(arch::Architecture, ::Val{N}) where {N} = heuristic_groupsize(arch.device, Val(N))
heuristic_groupsize(::Architecture{Kind,CPU}, N) where {Kind} = 256

end
