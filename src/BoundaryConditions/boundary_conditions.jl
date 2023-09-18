module BoundaryConditions

export apply_bcs!

export DirichletBC, HalfCell, FullCell
export ContinuousBC, DiscreteBC
export BoundaryFunction, DiscreteBoundaryFunction, ContinuousBoundaryFunction

using FastIce.Grids
using FastIce.Fields
using FastIce.Utils

using KernelAbstractions
using Adapt

include("utils.jl")
include("boundary_function.jl")
include("dirichlet_bc.jl")

function apply_bcs!(::Val{D}, backend, grid, fields, bcs; async=true) where {D}
    sizes   = ntuple(I -> remove_dim(Val(D), size(fields[I])), Val(length(fields)))
    ndrange = max.(sizes...)
    _apply_bcs!(backend, 256)(Val(D), grid, fields, sizes, bcs; ndrange)
    async || KernelAbstractions.synchronize(backend)
    return
end

@kernel function _apply_bcs!(dim, grid, fields, sizes, bcs)
    I = @index(Global, Cartesian)
    ntuple(Val(length(fields))) do ifield
        Base.@_inline_meta
        if _checkindices(sizes[ifield], I)
            _apply_bcs_lr!(dim, grid, fields[ifield], I, bcs[ifield]...)
        end
    end
end

@inline function _apply_bcs_lr!(dim, grid, f, I, lbc, rbc)
    loc = location(f)
    _apply_bc!(Val(1), dim, grid, f, loc, I, lbc)
    _apply_bc!(Val(2), dim, grid, f, loc, I, rbc)
    return
end

@inline _apply_bc!(side, dim, grid, f, loc, Ibc, ::Nothing) = nothing

end