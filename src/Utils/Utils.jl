module Utils

export remove_dim, insert_dim
export split_ndrange
export HideBoundaries, hide
export Pipeline
export extrapolate!

using FastIce.Fields
using FastIce.Architectures
using KernelAbstractions

include("pipelines.jl")
include("extrapolate.jl")
include("split_ndrange.jl")
include("hide_boundaries.jl")

# Returns a copy of the tuple `A` with element in position `D` removed
@inline remove_dim(::Val{D}, A::NTuple{N}) where {D,N} = ntuple(I -> I < D ? A[I] : A[I+1], Val(N - 1))
@inline remove_dim(::Val{1}, I::NTuple{1}) = 1

# Same as `remove_dim`, but for `CartesianIndex`
@inline remove_dim(dim, I::CartesianIndex) = remove_dim(dim, Tuple(I)) |> CartesianIndex

# Returns a copy of tuple `A`, but inserts `i` into position `D`
@inline insert_dim(::Val{D}, A::NTuple{N}, i) where {D,N} =
    ntuple(Val(N + 1)) do I
        if I < D
            A[I]
        elseif I == D
            i
        else
            A[I-1]
        end
    end

# Same as `insert_dim`, but for `CartesianIndex`
@inline insert_dim(dim, A::CartesianIndex, i) = insert_dim(dim, Tuple(A), i) |> CartesianIndex

end
