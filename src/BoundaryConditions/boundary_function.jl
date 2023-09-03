abstract type BoundaryFunction{F} end

struct ReducedDimensions end
struct FullDimensions    end

@inline _reduce(::Type{ReducedDimensions}, I, dim) = remove_dim(dim, I)
@inline _reduce(::Type{FullDimensions}, I, dim)    = I

struct ContinuousBoundaryFunction{F,P,RF} <: BoundaryFunction{F}
    fun::F
    parameters::P
    ContinuousBoundaryFunction{RF}(fun::F, params::P) where {RF,F,P} = new{F,P,RF}(fun, params)
end

struct DiscreteBoundaryFunction{F,P,RF} <: BoundaryFunction{F}
    fun::F
    parameters::P
    DiscreteBoundaryFunction{RF}(fun::F, params::P) where {F,P,RF} = new{F,P,RF}(fun, params)
end

const CBF{F,P,RF} = ContinuousBoundaryFunction{F,P,RF} where {F,P,RF}
const DBF{F,P,RF} = DiscreteBoundaryFunction{F,P,RF} where {F,P,RF}

const CDBF{F,P} = Union{CBF{F,P}, DBF{F,P}} where {F,P}

@inline _params(::CDBF{F,Nothing}) where {F} = ()
@inline _params(cbf::CDBF{F}) where {F} = cbf.parameters

@inline (bc::CBF{F,P,RF})(grid, loc, dim, I) where {F,P,RF} = bc.fun(_reduce(RF, coord(grid, loc, I), dim)..., _params(bc)...)

@inline (bc::DBF{F,P,RF})(grid, loc, dim, I) where {F,P,RF} =  bc.fun(grid, loc, dim, Tuple(_reduce(RF, I, dim))..., _params(bc)...)

# Create a continous or discrete boundary function
# if discrete = true, the function has signature f(grid, loc, dim, inds...)
# if reduce_dims = false, the boundary condition function accepts the same number of coordinates as the number of indices
function BoundaryFunction(fun::Function; discrete=false, parameters=nothing, reduce_dims=true)
    RF = reduce_dims ? ReducedDimensions : FullDimensions
    discrete ? DiscreteBoundaryFunction{RF}(fun, parameters) : ContinuousBoundaryFunction{RF}(fun, parameters)
end