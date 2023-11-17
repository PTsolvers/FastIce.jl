using KernelAbstractions
using Printf
using AMDGPU

@kernel function diffusion_kernel!(A_new, A, h, _dx2, _dy2, _dz2)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds if ix ∈ axes(A_new, 1)[2:end-1] && iy ∈ axes(A_new, 2)[2:end-1] && iz ∈ axes(A_new, 3)[2:end-1]
        A_new[ix, iy, iz] = A[ix, iy, iz] + h * ((A[ix-1, iy  , iz  ] + A[ix+1, iy  , iz  ] - 2.0 * A[ix, iy, iz]) * _dx2 +
                                                           (A[ix  , iy-1, iz  ] + A[ix  , iy+1, iz  ] - 2.0 * A[ix, iy, iz]) * _dy2 +
                                                           (A[ix  , iy  , iz-1] + A[ix  , iy  , iz+1] - 2.0 * A[ix, iy, iz]) * _dz2  )
    end
end

function compute_ka(backend, A_new, A, h, _dx2, _dy2, _dz2, iters)
    tic = time_ns()
    for _ = 1:iters
        diffusion_kernel!(backend, 256)(A_new, A, h, _dx2, _dy2, _dz2; ndrange=size(A))
        # diffusion_kernel!(backend, 256, size(A))(A_new, A, h, _dx2, _dy2, _dz2)
        KernelAbstractions.synchronize(backend)
        A, A_new = A_new, A
    end
    wtime = (time_ns() - tic) * 1e-9
    return wtime
end

function main(backend=CPU(), T::DataType=Float64)
    task_id = parse(Int, ENV["SLURM_LOCALID"])
    AMDGPU.device_id!(task_id + 1)
    # AMDGPU.device_id!(task_id*2 + 1)
    @show AMDGPU.device()
    # numerics
    iters, warmup = 35, 5
    nx, ny, nz = 1024, 1024, 1024
    _dx2 = _dy2 = _dz2 = 1.0
    h = 1.0 / 6.1 / 2
    # init arrays
    A_ini = rand(T, nx, ny, nz)
    A     = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    A_new = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    KernelAbstractions.copyto!(backend, A, A_ini)
    KernelAbstractions.synchronize(backend)
    A_new = copy(A)

    # warmup
    compute_ka(backend, A_new, A, h, _dx2, _dy2, _dz2, warmup)

    # time
    for _ in 1:10
        wtime = compute_ka(backend, A_new, A, h, _dx2, _dy2, _dz2, (iters - warmup))
        # report
        A_eff = 2 / 2^30 * (nx-2) * (ny-2) * (nz-2) * sizeof(Float64)
        wtime_it = wtime ./ (iters - warmup)
        T_eff = A_eff ./ wtime_it
        @printf("Executed %d steps in = %1.3e sec (@ T_eff = %1.2f GB/s - device %s) \n", (iters - warmup), wtime, round(T_eff, sigdigits=3), AMDGPU.device_id(AMDGPU.device()))
    end
    return
end

backend = ROCBackend()
T::DataType = Float64

main(backend, T)