module Distributed

using FastIce.Grids
using FastIce.Fields
using FastIce.Architectures
using FastIce.BoundaryConditions
import FastIce.BoundaryConditions: apply_boundary_conditions!

using MPI
using KernelAbstractions

export CartesianTopology
export global_rank, shared_rank, node_name, cartesian_communicator, shared_communicator, coordinates
export dimensions, global_size, node_size
export global_grid_size, local_grid
export neighbors, neighbor
export override_boundary_conditions
export gather!

export DistributedBoundaryConditions

struct DistributedMPI end

function Architectures.Architecture(backend::Backend, dims::NTuple{N,Int}, comm::MPI.Comm=MPI.COMM_WORLD) where {N}
    topo = CartesianTopology(dims; comm)
    device = get_device(backend, shared_rank(topo))
    return Architecture{DistributedMPI,typeof(backend),typeof(device),typeof(topo)}(backend, device, topo)
end

include("topology.jl")
include("boundary_conditions.jl")
include("gather.jl")

end
