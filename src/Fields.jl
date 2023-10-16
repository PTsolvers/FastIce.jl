module Fields

export AbstractField
export Field, interior
export location, data, halo, set!

using Adapt
using OffsetArrays
using KernelAbstractions

import FastIce.Grids: Location, CartesianGrid, coord

abstract type AbstractField{T,N,L} <: AbstractArray{T,N} end

Base.@pure location(::AbstractField{T,N,L}) where {T,N,L} = L.instance
Base.@pure location(::AbstractField{T,N,L}, ::Val{I}) where {T,N,L,I} = L.instance[I]
Base.@pure Base.eltype(::AbstractField{T}) where {T} = T
Base.@pure Base.ndims(::AbstractField{T,N}) where {T,N} = N

Base.IndexStyle(::AbstractField) = IndexCartesian()

struct Field{T,N,L,D,H,I} <: AbstractField{T,N,L}
    data::D
    halo::H
    indices::I
    Field{L}(data::D, halo::H, indices::I) where {L,D,H,I} = new{eltype(data),ndims(data),L,D,H,I}(data, halo, indices)
end

data_axis(sz, h) = (1-h[1]):(sz+h[2])

function make_data(backend, T, sz, halo_sz)
    total_halo_size = map(sum, halo_sz)
    array = KernelAbstractions.allocate(backend, T, sz .+ total_halo_size)
    field_axes = ntuple(I -> data_axis(sz[I], halo_sz[I]), Val(length(sz)))
    return OffsetArray(array, field_axes)
end

const HaloSize{N,I<:Integer} = NTuple{N,Tuple{I,I}}

function Field(backend::Backend, grid::CartesianGrid, T::DataType, loc::L, halo::HaloSize) where {L}
    sz = size(grid, loc)
    data = make_data(backend, T, sz, halo)
    indices = Base.OneTo.(sz)
    return Field{L}(data, halo, indices)
end

expand_axis_halo(::Nothing) = (0, 0)
expand_axis_halo(halo::Integer) = (halo, halo)
expand_axis_halo(halo::Tuple) = (something(halo[1], 0), something(halo[2], 0))

expand_halo(::Val{N}, halo::HaloSize{N}) where {N} = halo
expand_halo(::Val{N}, halo::Tuple) where {N} = ntuple(I -> expand_axis_halo(halo[I]), Val(length(halo)))
expand_halo(::Val{N}, halo::Integer) where {N} = ntuple(I -> (halo, halo), Val(N))
expand_halo(::Val{N}, halo::Nothing) where {N} = ntuple(I -> (0, 0), Val(N))

expand_loc(::Val{N}, loc::NTuple{N,Location}) where {N} = loc
expand_loc(::Val{N}, loc::Location) where {N} = ntuple(_ -> loc, Val(N))

function Field(backend::Backend, grid::CartesianGrid, loc::L, T::DataType=eltype(grid); halo::H=nothing) where {L,H}
    N = ndims(grid)
    return Field(backend, grid, T, expand_loc(Val(N), loc), expand_halo(Val(N), halo))
end

Base.checkbounds(f::Field, I...) = checkbounds(f.data, I...)
Base.checkbounds(f::Field, I::Union{CartesianIndex,AbstractArray{<:CartesianIndex}}) = checkbounds(f.data, I)

Base.checkbounds(::Type{Bool}, f::Field, I...) = checkbounds(Bool, f.data, I...)
Base.checkbounds(::Type{Bool}, f::Field, I::Union{CartesianIndex,AbstractArray{<:CartesianIndex}}) = checkbounds(Bool, f.data, I)

Base.size(f::Field) = length.(f.indices)
Base.parent(f::Field) = parent(f.data)
Base.axes(f::Field) = f.indices

Base.view(f::Field, I...) = view(f.data, I...)

data(f::Field) = f.data

halo(f::Field) = f.halo
halo(f::Field, dim::Integer) = f.halo[dim]
halo(f::Field, dim::Integer, side::Integer) = f.halo[dim][side]

Adapt.adapt_structure(to, f::Field{T,N,L}) where {T,N,L} = Field{L}(Adapt.adapt(to, f.data), f.halo, f.indices)

Base.@propagate_inbounds Base.getindex(f::Field, inds...) = getindex(f.data, inds...)
Base.@propagate_inbounds Base.setindex!(f::Field, val, inds...) = setindex!(f.data, val, inds...)

Base.@propagate_inbounds Base.firstindex(f::Field) = firstindex(f.data)
Base.@propagate_inbounds Base.firstindex(f::Field, dim) = firstindex(f.data, dim)
Base.@propagate_inbounds Base.lastindex(f::Field) = lastindex(f.data)
Base.@propagate_inbounds Base.lastindex(f::Field, dim) = lastindex(f.data, dim)

function interior_indices(f::Field)
    return ntuple(ndims(f)) do I
        (firstindex(parent(f), I)+f.halo[I][1]):(lastindex(parent(f), I)-f.halo[I][2])
    end
end

function interior(f::Field)
    return view(parent(f), interior_indices(f)...)
end

set!(f::Field, other::Field) = (copy!(interior(f), interior(other)); nothing)
set!(f::Field{T}, val::T) where {T<:Number} = (fill!(interior(f), val); nothing)
set!(f::Field, A::AbstractArray) = (copy!(interior(f), A); nothing)

@kernel function _set_continuous!(dst, grid, loc, fun::F, args...) where {F}
    I = @index(Global, Cartesian)
    dst[I] = fun(coord(grid, loc, I)..., args...)
end

@kernel function _set_discrete!(dst, grid, loc, fun::F, args...) where {F}
    I = @index(Global, Cartesian)
    dst[I] = fun(grid, loc, Tuple(I)..., args...)
end

function set!(f::Field{T,N}, grid::CartesianGrid{N}, fun::F; discrete=false, parameters=()) where {T,F,N}
    loc = location(f)
    dst = interior(f)
    if discrete
        _set_discrete!(get_backend(dst), 256, size(dst))(dst, grid, loc, fun, parameters...)
    else
        _set_continuous!(get_backend(dst), 256, size(dst))(dst, grid, loc, fun, parameters...)
    end
    return
end

end
