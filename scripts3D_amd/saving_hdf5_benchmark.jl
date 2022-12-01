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
    dims      = MPI.Dims_create!(nnodes,[0,0,0])
    comm_cart = MPI.Cart_create(MPI.COMM_WORLD,dims,[0,0,0],1)
    me        = MPI.Comm_rank(comm_cart)
    coords    = MPI.Cart_coords(comm_cart,me)
    comm_node = MPI.Comm_split_type(comm_cart, MPI.COMM_TYPE_SHARED, me)
    me_local  = MPI.Comm_rank(comm_node)
    node_name = MPI.Get_processor_name()
    println("process # $me_local on node # '$node_name' says 'Hi!'")
    A = rand(Float32,511,511,511)
    
    MPI.Finalize()
    return
end

main()