using AMDGPU
using KernelAbstractions
using MPI
using Printf
# using BenchmarkTools

# node_id = parse(Int, ENV["SLURM_NODEID"])
# task_id = parse(Int, ENV["SLURM_LOCALID"])

# println("node $node_id, gpu $task_id")

# AMDGPU.device_id!(task_id + 1)
# dev_id = AMDGPU.device_id(AMDGPU.device())
# @show AMDGPU.device()

@kernel function update_H!(H2, @Const(H), dc, dt, _dx2, _dy2, _dz2)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds H2[ix+1, iy+1, iz+1] = H[ix+1, iy+1, iz+1] + dt #= * dc * ((H[ix, iy+1, iz+1] - 2H[ix+1, iy+1, iz+1] + H[ix+2, iy+1, iz+1]) * _dx2 +
                                                                      (H[ix+1, iy, iz+1] - 2H[ix+1, iy+1, iz+1] + H[ix+1, iy+2, iz+1]) * _dy2 +
                                                                      (H[ix+1, iy+1, iz] - 2H[ix+1, iy+1, iz+1] + H[ix+1, iy+1, iz+2]) * _dz2) =#
end

function update_H_roc!(H2, H, dc, dt, _dx2, _dy2, _dz2)
    ix = (workgroupIdx().x - UInt32(1)) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - UInt32(1)) * workgroupDim().y + workitemIdx().y
    iz = (workgroupIdx().z - UInt32(1)) * workgroupDim().z + workitemIdx().z
    if (ix < size(H, 1) - 2 && iy < size(H, 2) - 2 && iz < size(H, 3) - 2)
        @inbounds H2[ix+1, iy+1, iz+1] = H[ix+1, iy+1, iz+1] + dt #= * dc * ((H[ix, iy+1, iz+1] - 2H[ix+1, iy+1, iz+1] + H[ix+2, iy+1, iz+1]) * _dx2 +
                                                                          (H[ix+1, iy, iz+1] - 2H[ix+1, iy+1, iz+1] + H[ix+1, iy+2, iz+1]) * _dy2 +
                                                                          (H[ix+1, iy+1, iz] - 2H[ix+1, iy+1, iz+1] + H[ix+1, iy+1, iz+2]) * _dz2) =#
    end
    return
end

function compute_ka(backend, nx, ny, nz, H2, H, dc, dt, _dx2, _dy2, _dz2, iters)
    tic = time_ns()
    for _ = 1:iters
        update_H!(backend, 256, (nx - 2, ny - 2, nz - 2))(H2, H, dc, dt, _dx2, _dy2, _dz2)
        KernelAbstractions.synchronize(backend)
    end
    wtime = (time_ns() - tic) * 1e-9
    return wtime
end

function compute_roc(nblocks, nthreads, H2, H, dc, dt, _dx2, _dy2, _dz2, iters)
    tic = time_ns()
    for _ = 1:iters
        AMDGPU.@sync @roc gridsize=nblocks groupsize=nthreads update_H_roc!(H2, H, dc, dt, _dx2, _dy2, _dz2)
    end
    wtime = (time_ns() - tic) * 1e-9
    return wtime
end

function run(backend=CPU(), dims=(0, 0, 0); nx=128, ny=128, nz=128, dtype=Float64)
    MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    dims = Tuple(MPI.Dims_create(nprocs, dims))
    # create MPI communicator
    comm = MPI.Cart_create(MPI.COMM_WORLD, dims)
    me = MPI.Comm_rank(comm)
    # create communicator for the node and select device
    comm_node = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, me)
    dev_id = MPI.Comm_rank(comm_node)
    @show device = AMDGPU.device_id!(dev_id + 1)
    dev_id = AMDGPU.device_id(AMDGPU.device())
    # physics
    dc = 1.0
    _dx2, _dy2, _dz2 = 1.0, 1.0, 1.0
    dt = 1.0 / (_dx2 * dc * 6.1)
    # allocate
    H = KernelAbstractions.zeros(backend, dtype, nx, ny, nz)
    # benchmark
    iters, warmup = 35, 5
    nthreads = (256, 1, 1)
    nblocks = cld.((nx, ny, nz), nthreads)
    H2 = copy(H)
    KernelAbstractions.synchronize(backend)
    println("Let's fucking gooooooooo!")

    # GC.gc()
    # GC.enable(false)

    # compute_ka(backend, nx, ny, nz, H2, H, dc, dt, _dx2, _dy2, _dz2, warmup)
    # wtime = compute_ka(backend, nx, ny, nz, H2, H, dc, dt, _dx2, _dy2, _dz2, (iters - warmup))

    compute_roc(nblocks, nthreads, H2, H, dc, dt, _dx2, _dy2, _dz2, warmup)
    wtime = compute_roc(nblocks, nthreads, H2, H, dc, dt, _dx2, _dy2, _dz2, (iters - warmup))

    # GC.enable(true)
    # GC.gc()

    # perf
    A_eff = 2 / 2^30 * (nx - 2) * (ny - 2) * (nz - 2) * sizeof(Float64)
    # A_eff = 2 / 2^30 * nx * ny * nz * sizeof(Float64)
    wtime_it = wtime / (iters - warmup)
    T_eff = A_eff / wtime_it
    @printf("Executed %d steps in = %1.3e sec (@ T_eff = %1.2f GB/s - device %s) \n", (iters - warmup), wtime, round(T_eff, sigdigits=3), dev_id)
    println("Done")
    MPI.Finalize()
    return
end

run(ROCBackend(), (0, 0, 0); nx=1024, ny=1024, nz=1024)