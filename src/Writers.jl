module Writers

export write_h5, write_xdmf

using FastIce.Grids
using FastIce.Fields

using HDF5
using LightXML

"""
write_h5(path, fields, grid_l::CartesianGrid, grid_g::CartesianGrid, args...)

Write `fields` in HDF5 format to a file in `path` using local grid `grid_l` and global grid `grid_g` information.
"""
function write_h5(path, fields, grid_l::CartesianGrid, grid_g::CartesianGrid, args...)
    if !HDF5.has_parallel() && (length(args) > 0)
        @warn("HDF5 has no parallel support.")
    end
    I = size(grid_l) == size(grid_g) ? CartesianIndices(size(grid_l)) : CartesianIndex( Int.(((origin(grid_l) .- origin(grid_g)) .÷ spacing(grid_l))) ) .+ CartesianIndices(size(grid_l))
    h5open(path, "w", args...) do io
        for (name, field) in fields
            dset = create_dataset(io, "/$name", datatype(eltype(field)), dataspace(size(grid_g)))
            dset[I.indices...] = Array(interior(field))
        end
    end
    return
end

"""
write_h5(path, fields, grid::CartesianGrid, args...)

Write `fields` in HDF5 format to a file in `path` using grid `grid` information.
"""
write_h5(path, fields, grid::CartesianGrid, args...) = write_h5(path, fields, grid, grid, args...)

"""
    write_xdmf(path, h5_names, fields, grid_l::CartesianGrid, grid_g::CartesianGrid, timesteps=Float64(0.0))

Write metadata in Xdmf format.
"""
function write_xdmf(path, h5_names, fields, grid_l::CartesianGrid, grid_g::CartesianGrid, timesteps=Float64(0.0))
    xdoc = XMLDocument()
    xroot = create_root(xdoc, "Xdmf")
    set_attribute(xroot, "Version", "3.0")

    xdomain     = new_child(xroot, "Domain")
    xcollection = new_child(xdomain, "Grid")
    set_attribute(xcollection, "GridType", "Collection")
    set_attribute(xcollection, "CollectionType", "Temporal")

    for (it, tt) in enumerate(timesteps)
        xgrid = new_child(xcollection, "Grid")
        set_attribute(xgrid, "GridType", "Uniform")
        xtopo = new_child(xgrid, "Topology")
        set_attribute(xtopo, "TopologyType", "3DCoRectMesh")
        set_attribute(xtopo, "Dimensions", join(reverse(size(grid_g)) .+ 1, ' '))

        xtime = new_child(xgrid, "Time")
        set_attribute(xtime, "Value", "$tt")

        xgeom = new_child(xgrid, "Geometry")
        set_attribute(xgeom, "GeometryType", "ORIGIN_DXDYDZ")

        xorig = new_child(xgeom, "DataItem")
        set_attribute(xorig, "Format", "XML")
        set_attribute(xorig, "NumberType", "Float")
        set_attribute(xorig, "Dimensions", "$(length(size(grid_g))) ")
        add_text(xorig, join(reverse(origin(grid_l)), ' '))

        xdr = new_child(xgeom, "DataItem")
        set_attribute(xdr, "Format", "XML")
        set_attribute(xdr, "NumberType", "Float")
        set_attribute(xdr, "Dimensions", "$(length(size(grid_g)))")
        add_text(xdr, join(reverse(spacing(grid_g)), ' '))

        h5_path = h5_names[it]
        for (name, _) in fields
            create_xdmf_attribute(xgrid, h5_path, name, grid_g)
        end
    end

    save_file(xdoc, path)
    return
end
write_xdmf(path, h5_names, fields, grid::CartesianGrid) = write_xdmf(path, h5_names, fields, grid, grid)
write_xdmf(path, h5_names, fields, grid::CartesianGrid, timesteps) = write_xdmf(path, h5_names, fields, grid, grid, timesteps)

function create_xdmf_attribute(xgrid, h5_path, name, grid_g::CartesianGrid)
    # TODO: solve type and precision
    xattr = new_child(xgrid, "Attribute")
    set_attribute(xattr, "Name", name)
    set_attribute(xattr, "Center", "Cell")
    xdata = new_child(xattr, "DataItem")
    set_attribute(xdata, "Format", "HDF")
    set_attribute(xdata, "NumberType", "Float")
    set_attribute(xdata, "Precision", "8")
    set_attribute(xdata, "Dimensions", join(reverse(size(grid_g)), ' '))
    add_text(xdata, "$(h5_path):/$name")
    return xattr
end

end
