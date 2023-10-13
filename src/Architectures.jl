module Architectures

export AbstractArchitecture

export SingleDeviceArchitecture

export launch!, set_device!, set_device_and_priority!, heuristic_groupsize
export synchronize

using FastIce.Grids

using KernelAbstractions
import KernelAbstractions.Kernel

abstract type AbstractArchitecture end

device(::AbstractArchitecture) = error("device function must be defined for architecture")
backend(::AbstractArchitecture) = error("backend function must be defined for architecture")

set_device!(arch::AbstractArchitecture) = set_device!(device(arch))

synchronize(arch::AbstractArchitecture) = KernelAbstractions.synchronize(backend(arch))

function set_device_and_priority!(arch::AbstractArchitecture, prio::Symbol)
    set_device!(arch)
    KernelAbstractions.priority!(backend(arch), prio)
    return
end

heuristic_groupsize(arch::AbstractArchitecture) = heuristic_groupsize(device(arch))

struct SingleDeviceArchitecture{B,D} <: AbstractArchitecture
    backend::B
    device::D
end

function SingleDeviceArchitecture(backend::Backend)
    device = set_device!(backend, 1)
    return SingleDeviceArchitecture(backend, device)
end

set_device!(::SingleDeviceArchitecture{CPU}) = nothing
set_device!(::CPU, id::Integer) = nothing

heuristic_groupsize(::SingleDeviceArchitecture{CPU}) = 256

device(arch::SingleDeviceArchitecture) = arch.device

backend(arch::SingleDeviceArchitecture) = arch.backend

function launch!(arch::SingleDeviceArchitecture, grid::CartesianGrid, kernel; kwargs...)
    worksize = size(grid, Vertex())
    launch!(arch, worksize, kernel; kwargs...)
end

function launch!(arch::SingleDeviceArchitecture, worksize::NTuple{N,Int}, kernel::Pair{Kernel,Args};
                 boundary_conditions=nothing, async=true) where {N,Args}
    fun, args = kernel

    groupsize = heuristic_groupsize(arch)

    fun(arch.backend, groupsize, worksize)(args...)
    isnothing(boundary_conditions) || apply_boundary_conditions!(boundary_conditions)

    async || synchronize(arch.backend)
    return
end

end
