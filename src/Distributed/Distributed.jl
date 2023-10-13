module Distributed

using FastIce.Architectures
using FastIce.Grids
import FastIce.BoundaryConditions: apply_boundary_conditions!
using MPI
using KernelAbstractions

export CartesianTopology

export global_rank, shared_rank, node_name, cartesian_communicator, shared_communicator

export dimensions, global_size, node_size

export global_grid_size, local_grid

struct DistributedArchitecture{C,T,R} <: AbstractArchitecture
    child_arch::C
    topology::T
end

function DistributedArchitecture(backend::Backend, dims::NTuple{N,Int}; comm=MPI.COMM_WORLD) where {N}
    topo = CartesianTopology(dims; comm)
    device = set_device!(backend, shared_rank(topo))
    child_arch = SingleDeviceArchitecture(backend, device)
    return DistributedArchitecture(child_arch, topo)
end

device(arch::DistributedArchitecture) = device(arch.child_arch)

include("topology.jl")
include("boundary_conditions.jl")

end
