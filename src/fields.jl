module Fields

export AbstractField
export Field, interior, interior_indices, halo_region
export location, location_instance, data, halo, set!

using Adapt
using KernelAbstractions

abstract type AbstractField{T,N,L} <: AbstractArray{T,N} end

import Base.@pure
import Base.@propagate_inbounds

@pure location(::AbstractField{T,N,L}) where {T,N,L} = L
@pure Base.eltype(::AbstractField{T}) where {T} = T
@pure Base.ndims(::AbstractField{T,N}) where {T,N} = N

Base.IndexStyle(::AbstractField) = IndexCartesian()

Base.@propagate_inbounds _get_halo_side(h::Union{Number, Nothing}, side) = something(h, 0)
Base.@propagate_inbounds _get_halo_side(h::Tuple, side) = something(h[side], 0)

struct Field{T,N,L,D,H} <: AbstractField{T,N,L}
    data::D
    halo::H
end

Field{L}(data::D, halo::H) where {L,D,H} = Field{eltype(D),ndims(data),L,D,H}(data, halo)

function Field(backend::B, ::Type{T}, grid::D, loc::L, halo::H) where {B,T,D,L,H}
    halo_size = ntuple(Val(ndims(grid))) do dim
        if !isnothing(halo)
            _get_halo_side(halo[dim], 1) + _get_halo_side(halo[dim], 2)
        else
            0
        end
    end
    data = KernelAbstractions.allocate(backend, T, size(grid, loc) .+ halo_size)
    return Field{T,ndims(grid),loc,typeof(data),H}(data, halo)
end

Field(backend::B, grid::D, loc::L, halo::H) where {B,D,L,H} = Field(backend, eltype(grid), grid, loc, halo)

data(f::Field) = f.data

halo(f::Field) = f.halo
halo(f::Field, dim) = f.halo[dim]
halo(f::Field, dim, side) = _get_halo_side(f.halo[dim], side)

Base.size(f::Field) = size(data(f))
Base.parent(f::Field) = data(f)

Adapt.adapt_structure(to, f::Field) = Adapt.adapt(to, f.data)

@propagate_inbounds Base.getindex(f::Field, inds...) = getindex(data(f), inds...)

interior_indices(f::Field{T,N,L,D,Nothing}, dim) where {T,N,L,D} = axes(data(f), dim)

function interior_indices(f::Field, dim)
    return UnitRange(firstindex(f, dim) + halo(f, dim, 1), lastindex(f, dim) - halo(f, dim, 2))
end

interior_indices(f::Field) = ntuple(dim -> interior_indices(f, dim), Val(ndims(f)))

interior(f::Field{T,N,L,D,Nothing}) where {T,N,L,D} = data(f)
interior(f::Field) = view(data(f), interior_indices(f)...)

function halo_indices(f::Field, dim, side)
    return if side == 1
        UnitRange(firstindex(f, dim), halo(f, dim, side))
    else
        UnitRange(lastindex(f, dim) - halo(f, dim, side), lastindex(f, dim))
    end
end

function halo_region(f::Field, dim, side)
    indices = ntuple(Val(ndims(f))) do I
        Base.@_inline_meta
        I == dim ? (:) : halo_indices(f, I, side)
    end
    return view(data(f), indices...)
end

function interior_and_halo(f::Field, dim)
    indices = ntuple(Val(ndims(f))) do I
        Base.@_inline_meta
        I == dim ? (:) : interior_indices(f, I)
    end
    return view(data(f), indices...)
end

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
    loc = location(f)
    dst = interior(f)
    _set!(get_backend(dst), 256)(dst, fun, grid, loc; ndrange=size(dst))
    return
end

end