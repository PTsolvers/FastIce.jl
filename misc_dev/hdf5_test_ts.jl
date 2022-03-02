using MPI
using HDF5
using LightXML

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
    A           = fill(float(myrank), sz)  # local data
    # filename    = "test.h5"
    dims_g      = (dims[1]*sz[1],dims[2]*sz[2],dims[3]*sz[3])

    or          = (-1,-2,-3)
    dr          = (0.1, 0.1, 0.1)

    ts = Float64[]; tt = 0.0; h5_names = String[]

    for it=1:5

        A .= A.*2.0

        MPI.Barrier(comm_cart)
        
        filename = "test_$it.h5"
        push!(ts,tt); push!(h5_names,filename)

        h5open(filename, "w", comm_cart, info) do ff
            # Create dataset
            dset = create_dataset(ff, "/data", datatype(eltype(A)), dataspace(dims_g))
            # Write local data
            ix = (coords[1]*sz[1] + 1):(coords[1]+1)*sz[1]
            iy = (coords[2]*sz[2] + 1):(coords[2]+1)*sz[2]
            iz = (coords[3]*sz[3] + 1):(coords[3]+1)*sz[3]
            dset[ix,iy,iz] = A
        end

        # finalize(info)
        # MPI.Barrier(comm_cart)

        # write XDMF
        if myrank == 0
            xdoc = XMLDocument()
            xroot = create_root(xdoc, "Xdmf")
            set_attribute(xroot, "Version","3.0")

            xdomain = new_child(xroot, "Domain")
            xcollection = new_child(xdomain, "Grid")
            set_attribute(xcollection, "GridType","Collection")
            set_attribute(xcollection, "CollectionType","Temporal")

            for (it,tt) ∈ enumerate(ts)
                xgrid   = new_child(xcollection, "Grid")
                set_attribute(xgrid, "GridType","Uniform")
                xtopo = new_child(xgrid, "Topology")
                set_attribute(xtopo, "TopologyType", "3DCoRectMesh")
                set_attribute(xtopo, "Dimensions", join(reverse(dims_g).+1,' '))

                xtime = new_child(xgrid, "Time")
                set_attribute(xtime, "Value", "$tt")

                xgeom = new_child(xgrid, "Geometry")
                set_attribute(xgeom, "GeometryType", "ORIGIN_DXDYDZ")

                xorig = new_child(xgeom, "DataItem")
                set_attribute(xorig, "Format", "XML")
                set_attribute(xorig, "NumberType", "Float")
                set_attribute(xorig, "Dimensions", "$(length(dims))")
                add_text(xorig, join(reverse(or), ' '))

                xdr   = new_child(xgeom, "DataItem")
                set_attribute(xdr, "Format", "XML")
                set_attribute(xdr, "NumberType", "Float")
                set_attribute(xdr, "Dimensions", "$(length(dims))")
                add_text(xdr, join(reverse(dr), ' '))


                xattr = new_child(xgrid, "Attribute")
                set_attribute(xattr, "Name", "data")
                set_attribute(xattr, "Center", "Cell")
                xdata = new_child(xattr, "DataItem")
                set_attribute(xdata, "Format", "HDF")
                set_attribute(xdata, "NumberType", "Float")
                set_attribute(xdata, "Precision", "8")
                set_attribute(xdata, "Dimensions", join(reverse(dims_g), ' '))

                add_text(xdata, "$(h5_names[it]):/data")

                save_file(xdoc, "test.xdmf3")
            end
        end

        tt += 0.1
        # MPI.Barrier(comm_cart)
    end

    if myrank == 0 println("Done") end

    MPI.Finalize()
end # testset mpio

main()