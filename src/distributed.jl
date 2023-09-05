module Distributed

export CartesianTopology

export global_rank, shared_rank, node_rank, cartesian_communicator, shared_communicator, global_size, local_grid

using FastIce.Grids

using MPI

struct CartesianTopology{N}
    nprocs::Int
    dims::NTuple{N,Int}
    global_rank::Int
    node_rank::Int
    shared_rank::Int
    cart_coords::NTuple{N,Int}
    comm::MPI.Comm
    cart_comm::MPI.Comm
    shared_comm::MPI.Comm
end

function CartesianTopology(dims::NTuple{N,Int}; comm = MPI.COMM_WORLD) where {N}
    nprocs = MPI.Comm_size(comm)
    dims = Tuple(MPI.Dims_create(nprocs, dims))
    cart_comm  = MPI.Cart_create(MPI.COMM_WORLD, dims)
    global_rank = MPI.Comm_rank(cart_comm)
    shared_comm = MPI.Comm_split_type(cart_comm, MPI.COMM_TYPE_SHARED, global_rank)
    shared_rank = MPI.Comm_rank(shared_comm)
    node_group = MPI.Comm_group(shared_comm)
    node_rank = MPI.Group_rank(node_group)
    cart_coords = Tuple(MPI.Cart_coords(cart_comm))

    return CartesianTopology{N}(nprocs, dims, global_rank, node_rank, shared_rank, cart_coords, comm, cart_comm, shared_comm)
end

global_rank(t::CartesianTopology) = t.global_rank

shared_rank(t::CartesianTopology) = t.shared_rank

node_rank(t::CartesianTopology) = t.node_rank

cartesian_communicator(t::CartesianTopology) = t.cart_comm

shared_communicator(t::CartesianTopology) = t.shared_comm

Base.size(t::CartesianTopology) = t.dims

global_size(t::CartesianTopology, local_size) =  t.dims .* local_size

function local_grid(g::CartesianGrid, t::CartesianTopology)
    local_extent = extent(g) ./ t.dims
    local_size   = size(g) .÷ t.dims
    local_origin = origin(g) .+ local_extent .* t.cart_coords

    return CartesianGrid(local_origin, local_extent, local_size)
end

end