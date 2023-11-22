struct Continuous end
struct Discrete end

struct FunctionField{T,N,L,CD,F,G,P} <: AbstractField{T,N,L}
    func::F
    grid::G
    parameters::P
    function FunctionField{CD,L}(func::F, grid::G, parameters::P) where {CD,L,F,G,P}
        N = ndims(grid)
        T = eltype(grid)
        return new{T,N,L,CD,F,G,P}(func, grid, parameters)
    end
end

function FunctionField(func::F, grid::G, loc; discrete=false, parameters=nothing) where {F,G}
    loc = expand_loc(Val(ndims(grid)), loc)
    L   = typeof(loc)
    CD  = discrete ? Discrete : Continuous
    return FunctionField{CD,L}(func, grid, parameters)
end

Base.size(f::FunctionField) = size(f.grid, location(f))

@inline func_type(::FunctionField{T,N,L,CD}) where {T,N,L,CD} = CD

@inline _params(::Nothing) = ()
@inline _params(p) = p

Base.@propagate_inbounds function call_func(func::F, grid, loc, I, params, ::Type{Continuous}) where {F}
    func(coord(grid, loc, I)..., params...)
end

Base.@propagate_inbounds function call_func(func::F, grid, loc, I, params, ::Type{Discrete}) where {F}
    func(grid, loc, I, params...)
end

Base.@propagate_inbounds Base.getindex(f::FunctionField{T,N}, inds::Vararg{Integer,N}) where {T,N} = getindex(f, CartesianIndex(inds))
Base.@propagate_inbounds function Base.getindex(f::FunctionField{T,N}, I::CartesianIndex{N}) where {T,N}
    call_func(f.func, f.grid, location(f), I, _params(f.parameters), func_type(f))
end

function Adapt.adapt_structure(to, f::FunctionField{T,N,L,CD}) where {T,N,L,CD}
    FunctionField{CD,L}(Adapt.adapt(to, f.func),
                        Adapt.adapt(to, f.grid),
                        Adapt.adapt(to, f.parameters))
end
