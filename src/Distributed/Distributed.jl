"""
    Distributed

Tools for performing parallel computations in a distributed environment.
Contains tools for non-blocking halo exchange using MPI.
Implements `BoundaryCondition` API to conveniently define communication as an operation that fills halo buffers.
Together with `HideBoundaries` from the `BoundaryConditions` module enables hiding MPI communication behind computations.
"""
module Distributed

export CartesianTopology
export global_rank, shared_rank, node_name, cartesian_communicator, shared_communicator, coordinates
export dimensions, global_size, node_size
export global_grid_size, local_grid
export neighbors, neighbor, has_neighbor
export gather!

export ExchangeInfo, DistributedBoundaryConditions
export DistributedMPI

using FastIce.Grids
using FastIce.Fields
using FastIce.Architectures
using FastIce.BoundaryConditions
import FastIce.BoundaryConditions: apply_boundary_conditions!

using MPI
using KernelAbstractions

"Trait structure used as a type parameter to indicate that the Architecture is a distributed MPI Architecture."
struct DistributedMPI end

include("topology.jl")
include("boundary_conditions.jl")
include("gather.jl")

end
