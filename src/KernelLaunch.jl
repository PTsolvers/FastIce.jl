module KernelLaunch

export launch!

using FastIce.Grids
using FastIce.Architectures
using FastIce.BoundaryConditions

using KernelAbstractions

"""
    launch!(arch::Architecture, grid::CartesianGrid, kernel::Pair{K,Args}; <keyword arguments>) where {K,Args}

Launch a KernelAbstraction `kernel` on a `grid` using the backend from `arch`.
Either `worksize` or `location` must be provided as keyword arguments.

# Keyword Arguments
- `worksize`: worksize of a kernel, i.e. how many grid points are included in each spatial direction.
- `location[=nothing]`: compute worksize as a size of the grid at a specified location.
    If only one location is provided, e.g. `location=Vertex()`, then this location will be used for all spacial directions.
- `offset[=nothing]`: index offset for all grid indices as a `CartesianIndex`.
- `expand[=nothing]`: if provided, the worksize is increased by `2*expand`, and offset is set to `-expand`, or combined with user-provided offset.
- `hide_boundaries[=nothing]`: instance of `HideBoundaries`, that will be used to overlap boundary processing with computations at inner points of the domain.
- `outer_width[=nothing]`: if `hide_boundaries` is specified, used to determine the decomposition of the domain into inner and outer regions.
- `boundary_conditions[=nothing]`: a tuple of boundary condition batches for each side of every spatial direction.
- `async[=true]`: if set to `false`, will block the host until the kernel is finished executing.
"""
function launch!(arch::Architecture, grid::CartesianGrid, kernel::Pair{K,Args};
                 worksize=nothing,
                 location=nothing,
                 offset=nothing,
                 expand=nothing,
                 boundary_conditions=nothing,
                 hide_boundaries=nothing,
                 outer_width=nothing,
                 async=true) where {K,Args}
    fun, args = kernel

    if isnothing(location) && isnothing(worksize)
        throw(ArgumentError("either grid location or worksize must me specified"))
    end

    if !isnothing(location) && !isnothing(worksize)
        throw(ArgumentError("either grid location or worksize must me specified, but not both"))
    end

    if isnothing(worksize)
        worksize = size(grid, location)
    end

    if !isnothing(expand)
        if !isnothing(offset)
            offset -= oneunit(CartesianIndex{ndims(grid)})
        else
            offset = -oneunit(CartesianIndex{ndims(grid)})
        end
        worksize = worksize .+ 2 .* expand
    end

    groupsize = heuristic_groupsize(arch, Val(length(worksize)))

    if isnothing(hide_boundaries)
        fun(backend(arch), groupsize, worksize)(args..., offset)
        isnothing(boundary_conditions) || apply_all_boundary_conditions!(arch, grid, boundary_conditions)
    else
        hide(hide_boundaries, arch, grid, boundary_conditions, worksize; outer_width) do indices
            sub_offset, ndrange = first(indices) - oneunit(first(indices)), size(indices)
            if !isnothing(offset)
                sub_offset += offset
            end
            fun(backend(arch), groupsize)(args..., sub_offset; ndrange)
        end
    end

    async || KernelAbstractions.synchronize(backend(arch))
    return
end

function apply_all_boundary_conditions!(arch::Architecture, grid::CartesianGrid{N}, boundary_conditions) where {N}
    ntuple(Val(N)) do D
        apply_boundary_conditions!(Val(1), Val(D), arch, grid, boundary_conditions[D][1])
        apply_boundary_conditions!(Val(2), Val(D), arch, grid, boundary_conditions[D][2])
    end
    return
end

end
