function extrapolate!(A::Field; async=true)
    backend = get_backend(A)
    I = ntuple(Val(ndims(A))) do dim
        halo(A, dim, 1) > 0 ? 0 : nothing
    end
    N = ntuple(Val(ndims(A))) do dim
        halo(A, dim, 2) > 0 ? size(A, dim) + 1 : nothing
    end
    kernel_extrapolate_x!(backend, 256, (size(A, 2) + 2, size(A, 3) + 2))(A, I[1], N[1])
    kernel_extrapolate_y!(backend, 256, (size(A, 1) + 2, size(A, 3) + 2))(A, I[2], N[2])
    kernel_extrapolate_z!(backend, 256, (size(A, 1) + 2, size(A, 2) + 2))(A, I[3], N[3])
    async || KernelAbstractions.synchronize(backend)
    return
end
