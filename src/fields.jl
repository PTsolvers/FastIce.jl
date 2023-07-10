module Fields

export AbstractField
export Field

abstract type AbstractField{T,N,L} <: AbstractArray{T,N} end

import Base.@pure

@pure location(::AbstractField{T,N,L}) where {T,N,L} = L
@pure location_instance(::AbstractField{T,N,L}) where {T,N,L} = L.instance
@pure Base.eltype(::AbstractField{T}) where T = T
@pure Base.ndims(::AbstractField{T,N}) where {T,N} = N

Base.IndexStyle(::AbstractField) = IndexCartesian()

struct Field{T,N,L,D,H} <: AbstractField{T,N,L}
    data::D
    halo::H
end

Field{L}(data::D, halo::H) where {L,D,H} = Field{D, ndims(data), L, D, H}(data, halo)

@inline data(f::Field) = f.data
@inline halo(f::Field) = f.halo

@inline Base.size(f::Field) = size(data(f))
@inline Base.parent(f::Field) = data(f)

Base.@propagate_inbounds Base.getindex(f::Field, inds...) = getindex(data(f), inds...)

@inline interior_indices(f::Field{T,N,L,D,Nothing}) where {T,N,L,D} = axes(data(f))
@inline interior_indices(f::Field) = UnitRange.(halo(f).+1,size(f).-halo(f))

@inline interior(f::Field{T,N,L,D,Nothing}) where {T,N,L,D} = data(f)

@inline interior(f::Field) = view(data(f), interior_indices(f)...)

end