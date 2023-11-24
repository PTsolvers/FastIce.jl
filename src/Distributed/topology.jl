"""
    CartesianTopology

Represents N-dimensional Cartesian topology of distributed processes.
"""
struct CartesianTopology{N}
    nprocs::Int
    dims::NTuple{N,Int}
    global_rank::Int
    shared_rank::Int
    cart_coords::NTuple{N,Int}
    neighbors::NTuple{N,NTuple{2,Int}}
    comm::MPI.Comm
    cart_comm::MPI.Comm
    shared_comm::MPI.Comm
    node_name::String
end

"""
    CartesianTopology(dims::NTuple{N,Int}; comm=MPI.COMM_WORLD) where {N}

Create an N-dimensional Cartesian topology with dimensions `dims` and using base MPI communicator `comm` (by default, `comm=MPI.COMM_WORLD`).
If all entries in `dims` are not equal to `0`, the product of `dims` should be equal to the total number of MPI processes `MPI.Comm_size(comm)`.
If any (or all) entries of `dims` are `0`, the dimensions in the corresponding spatial directions will be picked automatically.
"""
function CartesianTopology(dims::NTuple{N,Int}; comm=MPI.COMM_WORLD) where {N}
    nprocs      = MPI.Comm_size(comm)
    dims        = Tuple(MPI.Dims_create(nprocs, dims))
    cart_comm   = MPI.Cart_create(MPI.COMM_WORLD, dims)
    global_rank = MPI.Comm_rank(cart_comm)
    shared_comm = MPI.Comm_split_type(cart_comm, MPI.COMM_TYPE_SHARED, global_rank)
    shared_rank = MPI.Comm_rank(shared_comm)
    node_name   = MPI.Get_processor_name()
    cart_coords = Tuple(MPI.Cart_coords(cart_comm))

    neighbors = ntuple(Val(N)) do dim
        MPI.Cart_shift(cart_comm, dim - 1, 1)
    end

    return CartesianTopology{N}(nprocs, dims, global_rank, shared_rank, cart_coords, neighbors, comm, cart_comm, shared_comm, node_name)
end

"""
    global_rank(t::CartesianTopology)

Global id of a process in a Cartesian topology.
"""
global_rank(t::CartesianTopology) = t.global_rank

"""
    shared_rank(t::CartesianTopology)

Local id of a process within a single node. Can be used to set the GPU device.
"""
shared_rank(t::CartesianTopology) = t.shared_rank


"""
    node_name(t::CartesianTopology)

Name of a node according to `MPI.Get_processor_name()`.
"""
node_name(t::CartesianTopology) = t.node_name


"""
    cartesian_communicator(t::CartesianTopology)

MPI Cartesian communicator for the topology.
"""
cartesian_communicator(t::CartesianTopology) = t.cart_comm

"""
    shared_communicator(t::CartesianTopology)

MPI communicator for the processes sharing the same node.
"""
shared_communicator(t::CartesianTopology) = t.shared_comm


"""
    dimensions(t::CartesianTopology)

Dimensions of the topology as NTuple.
"""
dimensions(t::CartesianTopology) = t.dims

"""
    coordinates(t::CartesianTopology)

Coordinates of a current process within a Cartesian topology.
"""
coordinates(t::CartesianTopology) = t.cart_coords

"""
    neighbors(t::CartesianTopology)

Neighbors of a current process.

Returns NTuple containing process ids of the two immediate neighbors in each spatial direction, or MPI.PROC_NULL if no neighbor on a corresponding side.
"""
neighbors(t::CartesianTopology) = t.neighbors

"""
    neighbor(t::CartesianTopology, dim, side)

Returns id of a neighbor process in spatial direction `dim` on the side `side`, if this neighbor exists, or MPI.PROC_NULL otherwise.
"""
neighbor(t::CartesianTopology, dim, side) = t.neighbors[dim][side]


"""
    has_neighbor(t::CartesianTopology, dim, side)

Returns true if there a neighbor process in spatial direction `dim` on the side `side`, or false otherwise.
"""
has_neighbor(t::CartesianTopology, dim, side) = t.neighbors[dim][side] != MPI.PROC_NULL

"""
    global_size(t::CartesianTopology)

Total number of processes withing the topology.
"""
global_size(t::CartesianTopology) = MPI.Comm_size(t.cart_comm)

"""
    node_size(t::CartesianTopology)

Number of processes sharing the same node.
"""
node_size(t::CartesianTopology) = MPI.Comm_size(t.shared_comm)

"""
    global_grid_size(t::CartesianTopology, local_size)

Return the global size for a structured grid.
"""
global_grid_size(t::CartesianTopology, local_size) = t.dims .* local_size


"""
    local_grid(g::CartesianGrid, t::CartesianTopology)

Return a `CartesianGrid` covering the subdomain which corresponds to the current process.
"""
function local_grid(g::CartesianGrid, t::CartesianTopology)
    local_extent = extent(g) ./ t.dims
    local_size = size(g) .÷ t.dims
    local_origin = origin(g) .+ local_extent .* t.cart_coords

    return CartesianGrid(local_origin, local_extent, local_size)
end

"""
    Architecture(backend::Backend, topology::CartesianTopology) where {N}

Create a distributed Architecture using `backend` and `topology`. For GPU backends, device will be selected automatically based on a process id within a node.
"""
function Architectures.Architecture(backend::Backend, topology::CartesianTopology)
    device = get_device(backend, shared_rank(topology)+1)
    return Architecture{DistributedMPI,typeof(backend),typeof(device),typeof(topology)}(backend, device, topology)
end
