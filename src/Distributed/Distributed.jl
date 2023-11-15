"""
    Distributed

Tools for performing parallel computations in a distributed environment.
Contains tools for non-blocking halo exchange using MPI.
Implements `BoundaryCondition` API to conveniently define communication as an operation that fills halo buffers.
Together with `HideBoundaries` from the `BoundaryConditions` module enables hiding MPI communication behind computations.
"""
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
export neighbors, neighbor, has_neighbor
export gather!

export ExchangeInfo, DistributedBoundaryConditions

"Trait structure used as a type parameters to indicate that the Architecture is a distributed MPI Architecture."
struct DistributedMPI end

"""
    Architecture(backend::Backend, dims::NTuple{N,Int}, comm::MPI.Comm=MPI.COMM_WORLD) where {N}

Create a distributed Architecture using `backend`. For GPU backends, device will be selected automatically based on a process id within a node.
"""
function Architectures.Architecture(backend::Backend, dims::NTuple{N,Int}, comm::MPI.Comm=MPI.COMM_WORLD) where {N}
    topo = CartesianTopology(dims; comm)
    device = get_device(backend, shared_rank(topo)+1)
    return Architecture{DistributedMPI,typeof(backend),typeof(device),typeof(topo)}(backend, device, topo)
end

include("topology.jl")
include("boundary_conditions.jl")
include("gather.jl")

end
