flipped(loc::NTuple{N,Location}, ::Dim{D}) where {N,D} = ntuple(dim -> dim == D ? flip(loc[dim]) : loc[dim], Val(N))

Chmy.@add_cartesian function left(ω::AbstractMask{T,N}, loc::NTuple{N,Location}, dim::Dim{D}, I::Vararg{Integer,N}) where {T,N,D}
    Il = GridOperators.il(flip(loc[D]), loc[D], dim, I...)
    return at(ω, flipped(loc, dim), Il...)
end

Chmy.@add_cartesian function right(ω::AbstractMask{T,N}, loc::NTuple{N,Location}, dim::Dim{D}, I::Vararg{Integer,N}) where {T,N,D}
    Ir = GridOperators.ir(flip(loc[D]), loc[D], dim, I...)
    return at(ω, flipped(loc, dim), Ir...)
end

Base.@propagate_inbounds @generated function generic_isnullspace(ω::AbstractMask{T,N}, loc::NTuple{N,Location}, I::Vararg{Integer,N}) where {T,N}
    quote
        @inline
        Base.Cartesian.@nany $N D -> (left(ω, loc, Dim(D), I...) < 1e-6) || (right(ω, loc, Dim(D), I...) < 1e-6)
    end
end

Base.@propagate_inbounds function isnullspace(ω::AbstractMask{T,3}, loc::NTuple{3,Location}, I::Vararg{Integer,3}) where {T}
    return generic_isnullspace(ω, loc, I...)
end

Base.@propagate_inbounds function isnullspace(ω::AbstractMask{T,3}, loc::Tuple{Vertex,Vertex,Center}, I::Vararg{Integer,3}) where {T}
    return (left(ω, loc, Dim(1), I...) < 1e-6) || (right(ω, loc, Dim(1), I...) < 1e-6) ||
           (left(ω, loc, Dim(2), I...) < 1e-6) || (right(ω, loc, Dim(2), I...) < 1e-6)
end

Base.@propagate_inbounds function isnullspace(ω::AbstractMask{T,3}, loc::Tuple{Vertex,Center,Vertex}, I::Vararg{Integer,3}) where {T}
    return (left(ω, loc, Dim(1), I...) < 1e-6) || (right(ω, loc, Dim(1), I...) < 1e-6) ||
           (left(ω, loc, Dim(3), I...) < 1e-6) || (right(ω, loc, Dim(3), I...) < 1e-6)
end

Base.@propagate_inbounds function isnullspace(ω::AbstractMask{T,3}, loc::Tuple{Center,Vertex,Vertex}, I::Vararg{Integer,3}) where {T}
    return (left(ω, loc, Dim(2), I...) < 1e-6) || (right(ω, loc, Dim(2), I...) < 1e-6) ||
           (left(ω, loc, Dim(3), I...) < 1e-6) || (right(ω, loc, Dim(3), I...) < 1e-6)
end

Base.@propagate_inbounds isnullspace(ω::AbstractMask, loc, I::CartesianIndex) = isnullspace(ω, loc, Tuple(I)...)

Chmy.@add_cartesian isempty(ω::AbstractMask{T,N}, loc, I::Vararg{Integer,N}) where {T,N} = at(ω, loc, I...) < 1e-6

Chmy.@add_cartesian isvalid(ω::AbstractMask{T,N}, loc, I::Vararg{Integer,N}) where {T,N} = !(isnullspace(ω, loc, I...) || isempty(ω, loc, I...))
