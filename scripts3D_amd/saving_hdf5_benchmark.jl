using MPI,HDF5

function write_h5(path,fields,dim_g,I,args...)
    if !HDF5.has_parallel() && (length(args)>0)
        @warn("HDF5 has no parallel support.")
    end
    h5open(path, "w", args...) do io
        for (name,field) ∈ fields
            dset               = create_dataset(io, "/$name", datatype(eltype(field)), dataspace(dim_g))
            dset[I.indices...] = Array(field)
        end
    end
    return
end

function main()
    MPI.Init()
    nnodes    = MPI.Comm_size(MPI.COMM_WORLD)
    dims      = MPI.Dims_create!(nnodes,(0,0,0))
    comm_cart = MPI.Cart_create(MPI.COMM_WORLD,dims,(0,0,0),1)
    me        = MPI.Comm_rank(comm_cart)
    comm_node = MPI.Comm_split_type(comm_cart, MPI.MPI_COMM_TYPE_SHARED, me)
    me_local  = MPI.Comm_rank(comm_node)
    group     = MPI.Comm_group(comm_node)
    me_node   = MPI_group_rank(group)
    println("rank # $me , node # $me_node, local # $me_local")
    MPI.Finalize()
    return
end

main()