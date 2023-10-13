# Similar to `checkindex`, but for the multidimensional case. Custom axes aren't supported
@inline _checkindices(AI::Tuple, I::Tuple) = (I[1] <= AI[1]) && _checkindices(Base.tail(AI), Base.tail(I))
@inline _checkindices(AI::Tuple{}, I::Tuple{}) = true
@inline _checkindices(AI::Tuple, I::CartesianIndex) = _checkindices(AI, Tuple(I))

# For cell-centered fields, the boundary conditions are specified at index 0 (ghost cell)
@inline _get_i(::Center) = 0

# For vertex-centered fields, the boundary conditions are specified at index 1 (first inner point)
@inline _get_i(::Vertex) = 1

# Return 1D array index for the "left" and "right" sides of the array
@inline _index1(::Val{1}, L, sz) = _get_i(L)
@inline _index1(::Val{2}, L, sz) = -_get_i(L) + sz + 1

# Cartesian array index to store the boundary condition
@inline _bc_index(dim::Val{D}, side, loc, sz, Ibc) where {D} = insert_dim(dim, Ibc, _index1(side, loc[D], sz[D]))

# Return 1D offset for the "left" and "right" sides of the array to compute the flux projection
@inline _offset1(::Val{1}) = 1
@inline _offset1(::Val{2}) = -1

@inline _bc_offset(N, ::Val{D}, S) where {D} = ntuple(I -> I == D ? _offset1(S) : 0, N) |> CartesianIndex
