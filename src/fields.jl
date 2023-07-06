module Fields

export AbstractField
export Field

abstract type AbstractField{T,N,L} end

struct Field{T,N,L,D,H} <: AbstractField{T,N,L}
    data::D
    halo::H
end

@inline data(f::Field) = f.data
@inline halo(f::Field) = f.halo

import Base.@pure

@pure Base.eltype(::Field{T}) where T = T

@inline Base.size(f::Field) = size(data(f))

@inline interior_indices(f::Field{T,N,L,D,Nothing}) where {T,N,L,D} = axes(data(f))
@inline interior_indices(f::Field) = UnitRange.(halo(f).+1,size(f).-halo(f))

@inline interior(f::Field{T,N,L,D,Nothing}) where {T,N,L,D} = data(f)

@inline interior(f::Field) = view(data(f), interior_indices(f)...)

end