module Architecture

export AbstractArchitecture

export SingleDeviceArchitecture

export launch!, set_device!, heuristic_groupsize

using FastIce.Grids

using KernelAbstractions
import KernelAbstractions.Kernel

abstract type AbstractArchitecture end

set_device!(arch::AbstractArchitecture) = set_device!(device(arch))

heuristic_groupsize(arch::AbstractArchitecture) = heuristic_groupsize(device(arch))

struct SingleDeviceArchitecture{B,D} <: AbstractArchitecture
    backend::B
    device::D
end

set_device!(::SingleDeviceArchitecture{CPU}) = nothing

heuristic_groupsize(::SingleDeviceArchitecture{CPU}) = 256

device(arch::SingleDeviceArchitecture) = arch.device

function launch!(arch::SingleDeviceArchitecture, grid::CartesianGrid, kernel::Pair{Kernel,Args}; kwargs...) where {Args}
    worksize = size(grid, Vertex())
    launch!(arch, worksize, kernel; kwargs...)
end

function launch!(arch::SingleDeviceArchitecture, worksize::NTuple{N,Int}, kernel::Pair{Kernel,Args}; boundary_conditions=nothing, async=true) where {N,Args}
    fun, args = kernel

    groupsize = heuristic_groupsize(device(arch))

    fun(arch.backend, groupsize, worksize)(args...)
    isnothing(boundary_conditions) || apply_boundary_conditions!(boundary_conditions)

    async || synchronize(arch.backend)
    return
end

end