module Utils

export extrapolate!

using KernelAbstractions

@kernel function kernel_extrapolate_x!(A)
    iy, iz = @index(Global, NTuple)
    A[1  , iy + 1, iz + 1] = 2.0 * A[2      , iy + 1, iz + 1] - A[3      , iy + 1, iz + 1]
    A[end, iy + 1, iz + 1] = 2.0 * A[end - 1, iy + 1, iz + 1] - A[end - 2, iy + 1, iz + 1]
end

@kernel function kernel_extrapolate_y!(A)
    ix, iz = @index(Global, NTuple)
    A[ix + 1, 1  , iz + 1] = 2.0 * A[ix + 1, 2      , iz + 1] - A[ix + 1, 3      , iz + 1]
    A[ix + 1, end, iz + 1] = 2.0 * A[ix + 1, end - 1, iz + 1] - A[ix + 1, end - 2, iz + 1]
end

@kernel function kernel_extrapolate_z!(A)
    ix, iy = @index(Global, NTuple)
    A[ix + 1, iy + 1, 1  ] = 2.0 * A[ix + 1, iy + 1, 2      ] - A[ix + 1, iy + 1, 3      ]
    A[ix + 1, iy + 1, end] = 2.0 * A[ix + 1, iy + 1, end - 1] - A[ix + 1, iy + 1, end - 2]
end

function extrapolate!(A; async = true)
    backend = get_backend(A)
    kernel_extrapolate_x!(backend, 256, (size(A, 2) - 2, size(A, 3) - 2))(A)
    kernel_extrapolate_y!(backend, 256, (size(A, 1) - 2, size(A, 3) - 2))(A)
    kernel_extrapolate_z!(backend, 256, (size(A, 1) - 2, size(A, 2) - 2))(A)
    async || synchronize(backend)
    return
end

end