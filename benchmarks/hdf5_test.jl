using MPI
using HDF5

function main()
    MPI.Init()
    dims        = [2,2,2]
    comm        = MPI.COMM_WORLD
    nprocs      = MPI.Comm_size(comm)
    MPI.Dims_create!(nprocs, dims)
    comm_cart   = MPI.Cart_create(comm, dims, [0,0,0], 1)
    myrank      = MPI.Comm_rank(comm_cart)
    nprocs      = MPI.Comm_size(comm_cart)
    coords      = MPI.Cart_coords(comm_cart)
    info        = MPI.Info()
    sz          = (10,11,12)
    A           = fill(myrank, sz)  # local data
    filename    = "test.h5"
    h5open(filename, "w", comm_cart, info) do ff
        dims_g = (dims[1]*sz[1],dims[2]*sz[2],dims[3]*sz[3])
        # Create dataset
        dset = create_dataset(ff, "/data", datatype(eltype(A)), dataspace(dims_g))
        # Write local data
        ix = (coords[1]*sz[1] + 1):(coords[1]+1)*sz[1]
        iy = (coords[2]*sz[2] + 1):(coords[2]+1)*sz[2]
        iz = (coords[3]*sz[3] + 1):(coords[3]+1)*sz[3]
        dset[ix,iy,iz] = A
    end
    finalize(info)
    MPI.Barrier(comm_cart)
    MPI.Finalize()

    println("Done")
end # testset mpio

main()