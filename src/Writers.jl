module Writers

export write_h5, write_xdmf

using FastIce.Architectures
using FastIce.Distributed
using FastIce.Grids
using FastIce.Fields

using HDF5
using LightXML
using MPI

"""
    write_h5(arch::Architecture{DistributedMPI}, grid::CartesianGrid, path, fields)

Write output `fields` in HDF5 format to a file on `path` for global `grid` on distributed `arch`.
"""
function write_h5(arch::Architecture{DistributedMPI}, grid::CartesianGrid, path, fields)
    HDF5.has_parallel() || @warn("HDF5 has no parallel support.")
    topo = details(arch)
    comm = cartesian_communicator(topo)
    coords = coordinates(topo)
    sz = size(local_grid(grid, topo))
    c1 = coords .* sz .+ 1 |> CartesianIndex
    c2 = (coords .+ 1) .* sz |> CartesianIndex
    I = c1:c2
    h5open(path, "w", comm, MPI.Info()) do io
        write_dset(io, fields, size(grid), I.indices)
    end
    return
end

"""
    write_h5(arch::Architecture, grid::CartesianGrid, path, fields)

Write output `fields` in HDF5 format to a file on `path`.
"""
function write_h5(arch::Architecture, grid::CartesianGrid, path, fields)
    I = CartesianIndices(size(grid))
    h5open(path, "w") do io
        write_dset(io, fields, size(grid), I.indices)
    end
    return
end

function write_dset(io, fields, grid_size, inds)
    for (name, field) in fields
        dset = create_dataset(io, "/$name", datatype(eltype(field)), dataspace(grid_size))
        dset[inds...] = Array(interior(field))
    end
    return
end

"""
    write_xdmf(arch::Architecture{DistributedMPI}, grid::CartesianGrid, path, fields, h5_names, timesteps=Float64(0.0))

Write Xdmf metadata to `path` for corresponding `h5_names` and `fields` for global `grid` on distributed `arch`.
Saving time-dependant data can be achieved upon passing a vector to `h5_names` and `timesteps`.
"""
function write_xdmf(arch::Architecture{DistributedMPI}, grid::CartesianGrid, path, fields, h5_names, timesteps=Float64(0.0))
    topo = details(arch)
    grid_size = size(grid)
    grid_spacing = spacing(grid)
    grid_origin = origin(local_grid(grid, topo))

    xdoc = generate_xdmf(grid_size, grid_spacing, grid_origin, fields, h5_names, timesteps)

    save_file(xdoc, path)
    return
end

"""
    write_xdmf(arch::Architecture, grid::CartesianGrid, path, fields, h5_names, timesteps=Float64(0.0))

Write Xdmf metadata to `path` for corresponding `h5_names` and `fields`.
Saving time-dependant data can be achieved upon passing a vector to `h5_names` and `timesteps`.
"""
function write_xdmf(arch::Architecture, grid::CartesianGrid, path, fields, h5_names, timesteps=Float64(0.0))
    grid_size = size(grid)
    grid_spacing = spacing(grid)
    grid_origin = origin(grid)

    xdoc = generate_xdmf(grid_size, grid_spacing, grid_origin, fields, h5_names, timesteps)

    save_file(xdoc, path)
    return
end

function generate_xdmf(grid_size, grid_spacing, grid_origin, fields, h5_names, timesteps)
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
        set_attribute(xtopo, "Dimensions", join(reverse(grid_size) .+ 1, ' '))

        xtime = new_child(xgrid, "Time")
        set_attribute(xtime, "Value", "$tt")

        xgeom = new_child(xgrid, "Geometry")
        set_attribute(xgeom, "GeometryType", "ORIGIN_DXDYDZ")

        xorig = new_child(xgeom, "DataItem")
        set_attribute(xorig, "Format", "XML")
        set_attribute(xorig, "NumberType", "Float")
        set_attribute(xorig, "Dimensions", "$(length(grid_size)) ")
        add_text(xorig, join(reverse(grid_origin), ' '))

        xdr = new_child(xgeom, "DataItem")
        set_attribute(xdr, "Format", "XML")
        set_attribute(xdr, "NumberType", "Float")
        set_attribute(xdr, "Dimensions", "$(length(grid_size))")
        add_text(xdr, join(reverse(grid_spacing), ' '))

        h5_path = h5_names[it]
        for (name, _) in fields
            create_xdmf_attribute(xgrid, h5_path, name, grid_size)
        end
    end
    return xdoc
end

function create_xdmf_attribute(xgrid, h5_path, name, grid_size)
    # TODO: solve type and precision
    xattr = new_child(xgrid, "Attribute")
    set_attribute(xattr, "Name", name)
    set_attribute(xattr, "Center", "Cell")
    xdata = new_child(xattr, "DataItem")
    set_attribute(xdata, "Format", "HDF")
    set_attribute(xdata, "NumberType", "Float")
    set_attribute(xdata, "Precision", "8")
    set_attribute(xdata, "Dimensions", join(reverse(grid_size), ' '))
    add_text(xdata, "$(h5_path):/$name")
    return xattr
end

end
