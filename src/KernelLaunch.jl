module KernelLaunch

using FastIce.Grids
using FastIce.Architectures
using FastIce.BoundaryConditions

export launch!

function launch!(arch::Architecture, grid::CartesianGrid, kernel::Pair{K,Args};
                 location=nothing,
                 worksize=nothing,
                 offset=nothing,
                 expand=nothing,
                 hide_boundaries=nothing,
                 outer_width=nothing,
                 boundary_conditions=nothing,
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

    groupsize = heuristic_groupsize(arch, length(worksize))

    if isnothing(hide_boundaries)
        fun(arch.backend, groupsize, worksize)(args..., offset)
        isnothing(boundary_conditions) || apply_all_boundary_conditions!(arch, grid, boundary_conditions)
    else
        hide(hide_boundaries, arch, grid, boundary_conditions, worksize; outer_width) do indices
            sub_offset, ndrange = first(indices) - oneunit(first(indices)), size(indices)
            if !isnothing(offset)
                sub_offset += offset
            end
            fun(arch.backend, groupsize)(args..., sub_offset; ndrange)
        end
    end

    async || synchronize(arch.backend)
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
