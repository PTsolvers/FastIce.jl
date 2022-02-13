function write_h5(path,fields,comm,info,dim_g,I)
    h5open(path, "w", comm_cart, info) do io
        for name,field ∈ fields
            dset    = create_dataset(io, "/$name", datatype(eltype(field)), dataspace(dim_g))
            dset[I] = Array(field)
        end
    end
end

function write_xdmf(path,h5_path,fields,origin,spacing,dim_g)
    ox,oy,oz = origin
    dx,dy,dz = spacing
    xdoc = XMLDocument()
    xroot = create_root(xdoc, "Xdmf")
    set_attribute(xroot, "Version","3.0")

    xdomain = new_child(xroot, "Domain")
    xgrid   = new_child(xdomain, "Grid")
    set_attribute(xgrid, "GridType","Uniform")
    xtopo = new_child(xgrid, "Topology")
    set_attribute(xtopo, "TopologyType", "3DCoRectMesh")
    set_attribute(xtopo, "Dimensions", join(reverse(dim_g).+1,' '))

    xgeom = new_child(xgrid, "Geometry")
    set_attribute(xgeom, "GeometryType", "ORIGIN_DXDYDZ")

    xorig = new_child(xgeom, "DataItem")
    set_attribute(xorig, "Format", "XML")
    set_attribute(xorig, "NumberType", "Float")
    set_attribute(xorig, "Dimensions", "$(length(dim_g)) ")
    add_text(xorig, join((oz,oy,ox), ' '))

    xdr = new_child(xgeom, "DataItem")
    set_attribute(xdr, "Format", "XML")
    set_attribute(xdr, "NumberType", "Float")
    set_attribute(xdr, "Dimensions", "$(length(dim_g))")
    add_text(xdr, join((dz,dy,dx), ' '))

    for name,_ ∈ fields
        create_xdmf_attribute(xgrid,h5_path,name,dim_g)
    end

    save_file(xdoc, path)
end

function create_xdmf_attribute(xgrid,h5_path,name,dim_g)
    # TODO: solve type and precision
    xattr = new_child(xgrid, "Attribute")
    set_attribute(xattr, "Name", name)
    set_attribute(xattr, "Center", "Cell")
    xdata = new_child(xattr, "DataItem")
    set_attribute(xdata, "Format", "HDF")
    set_attribute(xdata, "NumberType", "Float")
    set_attribute(xdata, "Precision", "8")
    set_attribute(xdata, "Dimensions", join(reverse(dim_g), ' '))
    add_text(xdata, "$h5_path:/$name")
    return xattr
end