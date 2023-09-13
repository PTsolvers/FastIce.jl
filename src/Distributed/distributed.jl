module Distributed

using FastIce.Architecture
using FastIce.Grids

export CartesianTopology

export global_rank, shared_rank, node_name, cartesian_communicator, shared_communicator

export dimensions, global_size, node_size

export global_grid_size, local_grid

export split_ndrange

using FastIce.Grids

using MPI

include("topology.jl")

include("split_ndrange.jl")

struct DistributedArchitecture{C,T,R} <: AbstractArchitecture
    child_arch::C
    topology::T
    ranges::R
end

device(arch::DistributedArchitecture) = device(arch.child_arch)

function launch!(arch::DistributedArchitecture, grid::CartesianGrid, kernel::Pair{Kernel,Args}; boundary_conditions=nothing, async=true) where {Args}
    fun, args = kernel

    worksize = size(grid, Vertex())
    groupsize = heuristic_groupsize(arch.child_arch)

    fun(arch.backend, groupsize)(args...; ndrange=size(arch.ranges[end]), offset=first(arch.ranges[end]))


    isnothing(boundary_conditions) || apply_boundary_conditions!(boundary_conditions)

    async || synchronize(arch.backend)
    return
end

end