module Fields

export AbstractField
export Field, interior, interior_indices

using Adapt

abstract type AbstractField{T,N,L} <: AbstractArray{T,N} end

import Base.@pure
import Base.@propagate_inbounds

@pure location(::AbstractField{T,N,L}) where {T,N,L} = L
@pure location_instance(::AbstractField{T,N,L}) where {T,N,L} = L.instance
@pure Base.eltype(::AbstractField{T}) where {T} = T
@pure Base.ndims(::AbstractField{T,N}) where {T,N} = N

Base.IndexStyle(::AbstractField) = IndexCartesian()

struct Field{T,N,L,D,H} <: AbstractField{T,N,L}
    data::D
    halo::H
end

Field{L}(data::D, halo::H) where {L,D,H} = Field{eltype(D),ndims(data),L,D,H}(data, halo)

data(f::Field) = f.data
halo(f::Field) = f.halo

Base.size(f::Field) = size(data(f))
Base.parent(f::Field) = data(f)

Adapt.adapt_structure(to, f::Field) = Adapt.adapt(to, f.data)

@propagate_inbounds Base.getindex(f::Field, inds...) = getindex(data(f), inds...)

interior_indices(f::Field{T,N,L,D,Nothing}) where {T,N,L,D} = axes(data(f))
interior_indices(f::Field) = UnitRange.(halo(f) .+ 1, size(f) .- halo(f))

interior(f::Field{T,N,L,D,Nothing}) where {T,N,L,D} = data(f)

interior(f::Field) = view(data(f), interior_indices(f)...)

set!(f::Field, other::Field) = (copy!(interior(f), interior(other)); nothing)
set!(f::Field{T}, val::T) where {T} = (fill!(interior(f), val); nothing)
set!(f::Field, A::AbstractArray) = (copy!(interior(f), A); nothing)

import FastIce.Grids: CartesianGrid, coord

using KernelAbstractions

@kernel function _set!(dst, fun, grid, loc)
    I = @index(Global, Cartesian)
    dst[I] = fun(coord(grid, loc, I))
end

function set!(f::Field{T,N}, fun::F, grid::CartesianGrid{N}) where {T,N,F}
    loc = location_instance(f)
    dst = interior(f)
    _set!(get_backend(dst), 256)(dst, fun, grid, loc; ndrange=size(dst))
    return
end

end