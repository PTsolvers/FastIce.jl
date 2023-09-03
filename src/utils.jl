module Utils

export remove_dim, insert_dim

# Returns a copy of the tuple `A` with element in position `D` removed
@inline remove_dim(::Val{D}, A::NTuple{N}) where {D,N} = ntuple(I -> I < D ? A[I] : A[I+1], Val(N - 1))
@inline remove_dim(::Val{1}, I::NTuple{1}) = 1

# Same as `remove_dim`, but for `CartesianIndex`
@inline remove_dim(dim, I::CartesianIndex) = remove_dim(dim, Tuple(I)) |> CartesianIndex

# Returns a copy of tuple `A`, but inserts `i` into position `D`
@inline insert_dim(::Val{D}, A::NTuple{N}, i) where {D, N} = ntuple(Val(N + 1)) do I
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

export extrapolate!

using FastIce.Fields

using KernelAbstractions

@kernel function kernel_extrapolate_x!(A, I, N)
    iy, iz = @index(Global, NTuple)
    iy -= 1; iz -= 1
    @inbounds if !isnothing(I) A[I, iy, iz] = 2.0 * A[I+1, iy, iz] - A[I+2, iy, iz] end
    @inbounds if !isnothing(N) A[N, iy, iz] = 2.0 * A[N-1, iy, iz] - A[N-2, iy, iz] end
end

@kernel function kernel_extrapolate_y!(A, I, N)
    ix, iz = @index(Global, NTuple)
    ix -= 1; iz -= 1
    @inbounds if !isnothing(I) A[ix, I, iz] = 2.0 * A[ix, I+1, iz] - A[ix, I+2, iz] end
    @inbounds if !isnothing(N) A[ix, N, iz] = 2.0 * A[ix, N-1, iz] - A[ix, N-2, iz] end
end

@kernel function kernel_extrapolate_z!(A, I, N)
    ix, iy = @index(Global, NTuple)
    ix -= 1; iy -= 1
    @inbounds if !isnothing(I) A[ix, iy, I] = 2.0 * A[ix, iy, I+1] - A[ix, iy, I+2] end
    @inbounds if !isnothing(N) A[ix, iy, N] = 2.0 * A[ix, iy, N-1] - A[ix, iy, N-2] end
end

function extrapolate!(A::Field; async=true)
    backend = get_backend(A)
    I = ntuple(Val(ndims(A))) do dim
        halo(A, dim, 1) > 0 ? 0 : nothing
    end
    N = ntuple(Val(ndims(A))) do dim
        halo(A, dim, 2) > 0 ? size(A, dim) + 1 : nothing
    end
    kernel_extrapolate_x!(backend, 256, (size(A, 2)+2, size(A, 3)+2))(A, I[1], N[1])
    kernel_extrapolate_y!(backend, 256, (size(A, 1)+2, size(A, 3)+2))(A, I[2], N[2])
    kernel_extrapolate_z!(backend, 256, (size(A, 1)+2, size(A, 2)+2))(A, I[3], N[3])
    async || synchronize(backend)
    return
end

end