using KernelAbstractions
using MPI
using Printf

using AMDGPU

# using CUDA
# using NVTX

include("mpi_utils.jl")
include("mpi_utils2.jl")

macro inn(A) esc(:( $A[ix+1, iy+1, iz+1] )) end
macro d2_xi(A) esc(:( $A[ix+2, iy+1, iz+1] - $A[ix+1, iy+1, iz+1] - $A[ix+1, iy+1, iz+1] - $A[ix, iy+1, iz+1] )) end
macro d2_yi(A) esc(:( $A[ix+1, iy+2, iz+1] - $A[ix+1, iy+1, iz+1] - $A[ix+1, iy+1, iz+1] - $A[ix+1, iy, iz+1] )) end
macro d2_zi(A) esc(:( $A[ix+1, iy+1, iz+2] - $A[ix+1, iy+1, iz+1] - $A[ix+1, iy+1, iz+1] - $A[ix+1, iy+1, iz] )) end

@kernel function diffusion_kernel!(A_new, A, h, _dx, _dy, _dz, offset)
    ix, iy, iz = @index(Global, NTuple)
    ix += offset[1] - 1
    iy += offset[2] - 1
    iz += offset[3] - 1
    if ix ∈ axes(A_new, 1)[2:end-1] && iy ∈ axes(A_new, 2)[2:end-1] && iz ∈ axes(A_new, 3)[2:end-1]
        @inbounds A_new[ix, iy, iz] = A[ix, iy, iz] + h * ((A[ix-1, iy  , iz  ] + A[ix+1, iy  , iz  ] - 2.0 * A[ix, iy, iz]) * _dx * _dx +
                                                           (A[ix  , iy-1, iz  ] + A[ix  , iy+1, iz  ] - 2.0 * A[ix, iy, iz]) * _dy * _dy +
                                                           (A[ix  , iy  , iz-1] + A[ix  , iy  , iz+1] - 2.0 * A[ix, iy, iz]) * _dz * _dz  )
    end
    # if (ix < size(A, 1) - 2 && iy < size(A, 2) - 2 && iz < size(A, 3) - 2)
    #     # @inbounds @inn(A_new) = @inn(A) + h
    #     @inbounds @inn(A_new) = @inn(A) + h * (_dx * _dx * @d2_xi(A) + _dy * _dy * @d2_yi(A) + _dz * _dz * @d2_zi(A))
    # end
end

function lapl!(A_new, A, h, _dx, _dy, _dz)
    ix = (workgroupIdx().x - UInt32(1)) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - UInt32(1)) * workgroupDim().y + workitemIdx().y
    iz = (workgroupIdx().z - UInt32(1)) * workgroupDim().z + workitemIdx().z
    # if ix ∈ axes(A_new, 1)[2:end-1] && iy ∈ axes(A_new, 2)[2:end-1] && iz ∈ axes(A_new, 3)[2:end-1]
    #     @inbounds A_new[ix, iy, iz] = A[ix, iy, iz] + h #= * (_dx * _dx * @d2_xi(A) + _dy * _dy * @d2_yi(A) + _dz * _dz * @d2_zi(A)) =#
    # end
    if (ix < size(A, 1) - 2 && iy < size(A, 2) - 2 && iz < size(A, 3) - 2)
        @inbounds @inn(A_new) = @inn(A) + h #= * (_dx * _dx * @d2_xi(A) + _dy * _dy * @d2_yi(A) + _dz * _dz * @d2_zi(A)) =#
    end
    return
end

function compute_ka(hide_comm, backend, neighbors, ranges, A_new, A, h, _dx, _dy, _dz, iters, me)
    (me==0) && print("Starting the time loop 🚀...")
    tic = time_ns()
    for _ = 1:iters
        # copyto!(A, A_new)
        # AMDGPU.synchronize(blocking=false) #KernelAbstractions.synchronize(backend)
        # hide_comm(diffusion_kernel!(backend, 256), neighbors, ranges, A_new, A, h, _dx, _dy, _dz)
        # A, A_new = A_new, A

        diffusion_kernel!(backend, 256)(A_new, A, h, _dx, _dy, _dz, (1, 1, 1); ndrange=size(A))
        # diffusion_kernel!(backend, 256)(A_new, A, h, _dx, _dy, _dz, (1, 1, 1); ndrange=size(A) .- 2)
        # diffusion_kernel!(backend, 256, (size(A) .- 2))(A_new, A, h, _dx, _dy, _dz, (1, 1, 1))
        AMDGPU.synchronize(blocking=false) #KernelAbstractions.synchronize(backend)
        # A, A_new = A_new, A
    end
    wtime = (time_ns() - tic) * 1e-9
    (me==0) && println("done")
    return wtime
end

function compute_roc(A_new, A, h, _dx, _dy, _dz, iters, nblocks, nthreads, me)
    (me==0) && print("Starting the time loop 🚀...")
    tic = time_ns()
    for _ = 1:iters
        AMDGPU.@sync @roc gridsize=nblocks groupsize=nthreads lapl!(A_new, A, h, _dx, _dy, _dz)
        # A, A_new = A_new, A
    end
    wtime = (time_ns() - tic) * 1e-9
    (me==0) && println("done")
    return wtime
end

function main(backend=CPU(), T::DataType=Float64, dims=(0, 0, 0))
    # physics
    l = 10.0
    # numerics
    iters, warmup = 35, 5
    nx, ny, nz = 1024, 1024, 1024
    b_width = (128, 8, 4)
    nthreads = (256, 1, 1)
    nblocks = cld.((nx, ny, nz), nthreads)
    dims, comm, me, neighbors, coords, device = init_distributed(dims; init_MPI=true)
    dx, dy, dz = l ./ (nx, ny, nz)
    _dx, _dy, _dz = 1.0 ./ (dx, dy, dz)
    h = min(dx, dy ,dz)^2 / 6.1
    # init arrays
    x_g = (ix, dx) -> (coords[1] * (nx - 2) + (ix-1)) * dx + dx/2
    y_g = (iy, dy) -> (coords[2] * (ny - 2) + (iy-1)) * dy + dy/2
    z_g = (iz, dz) -> (coords[3] * (nz - 2) + (iz-1)) * dz + dz/2
    # Gaussian
    # A_ini = zeros(T, nx, ny, nz)
    # A_ini .= [exp(-(x_g(ix, dx) - l/2)^2 - (y_g(iy, dy) - l/2)^2 - (z_g(iz, dz) - l/2)^2) for ix=1:size(A_ini, 1), iy=1:size(A_ini, 2), iz=1:size(A_ini, 3)]
    A_ini = rand(T, nx, ny, nz)

    A     = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    A_new = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    KernelAbstractions.copyto!(backend, A, A_ini)
    AMDGPU.synchronize(blocking=false) #KernelAbstractions.synchronize(backend)
    A_new = copy(A)

    ### to be hidden later
    ranges = split_ndrange(A, b_width)

    exchangers = ntuple(Val(length(neighbors))) do _
        ntuple(_ -> Exchanger(backend, device), Val(2))
    end

    function hide_comm(f, neighbors, ranges, args...)
        f(args..., first(ranges[end]); ndrange=size(ranges[end]))
        for dim in reverse(eachindex(neighbors))
            ntuple(Val(2)) do side
                rank   = neighbors[dim][side]
                halo   = get_recv_view(Val(side), Val(dim), A_new)
                border = get_send_view(Val(side), Val(dim), A_new)
                range  = ranges[2*(dim-1) + side]
                offset, ndrange = first(range), size(range)
                start_exchange(exchangers[dim][side], comm, rank, halo, border) do compute_bc
                    f(args..., offset; ndrange)
                    if compute_bc
                        # apply_bcs!(Val(dim), fields, bcs.velocity)
                    end
                    AMDGPU.synchronize(blocking=false) #KernelAbstractions.synchronize(backend)
                end
            end
            wait.(exchangers[dim])
        end
        AMDGPU.synchronize(blocking=false) #KernelAbstractions.synchronize(backend)
    end
    ### to be hidden later

    # GC.gc()
    # GC.enable(false)

    compute_ka(hide_comm, backend, neighbors, ranges, A_new, A, h, _dx, _dy, _dz, warmup, me)
    wtime = compute_ka(hide_comm, backend, neighbors, ranges, A_new, A, h, _dx, _dy, _dz, (iters - warmup), me)

    # compute_roc(A_new, A, h, _dx, _dy, _dz, warmup, nblocks, nthreads, me)
    # wtime = compute_roc(A_new, A, h, _dx, _dy, _dz, (iters - warmup), nblocks, nthreads, me)

    # GC.enable(true)
    # GC.gc()

    # perf
    A_eff = 2 / 2^30 * (nx-2) * (ny-2) * (nz-2) * sizeof(Float64)
    wtime_it = wtime / (iters - warmup)
    T_eff = A_eff / wtime_it
    # (me==0) && @printf("Executed %d steps in = %1.3e sec (@ T_eff = %1.2f GB/s) \n", (iters - warmup), wtime, round(T_eff, sigdigits=3))
    @printf("Executed %d steps in = %1.3e sec (@ T_eff = %1.2f GB/s - device %s) \n", (iters - warmup), wtime, round(T_eff, sigdigits=3), AMDGPU.device_id(AMDGPU.device()))

    finalize_distributed(; finalize_MPI=true)
    return
end

backend = ROCBackend()
T::DataType = Float64
dims = (0, 0, 0)

main(backend, T, dims)
# main()