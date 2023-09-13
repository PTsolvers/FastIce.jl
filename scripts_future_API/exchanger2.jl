using KernelAbstractions
using MPI
using CUDA
using NVTX

include("mpi_utils.jl")
include("mpi_utils2.jl")

@kernel function do_work!(A, me, offset)
    ix, iy, iz = @index(Global, NTuple)
    ix += offset[1] - 1
    iy += offset[2] - 1
    iz += offset[3] - 1
    for _ in 1:10
        # if (ix > 1 && ix < size(A, 1)) &&
        #    (iy > 1 && iy < size(A, 2)) &&
        #    (iz > 1 && iz < size(A, 3))
            A[ix, iy, iz] = me
        # end
    end
end

function main(backend = CPU(), T::DataType = Float64, dims = (0, 0, 0))

    # numerics
    dims, comm, me, neighbors, coords, device = init_distributed(dims; init_MPI=true)

    nx, ny, nz = 6, 6, 6
    b_width = (2, 2, 2)

    # init arrays
    A = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    fill!(A, -1)

    ranges = split_ndrange(A, b_width)

    exchangers = ntuple(Val(length(neighbors))) do _
        ntuple(_ -> Exchanger(backend, device), Val(2))
    end

    do_work!(backend, 256)(A, me, first(ranges[end]); ndrange=size(ranges[end]))

    for dim in reverse(eachindex(neighbors))
        ntuple(Val(2)) do side
            rank   = neighbors[dim][side]
            halo   = get_recv_view(Val(side), Val(dim), A)
            border = get_send_view(Val(side), Val(dim), A)
            range  = ranges[2*(dim-1) + side]
            offset, ndrange = first(range), size(range)
            start_exchange(exchangers[dim][side], comm, rank, halo, border) do compute_bc
                NVTX.@range "borders" do_work!(backend, 256)(A, me, offset; ndrange)
                if compute_bc
                    # apply_bcs!(Val(dim), fields, bcs.velocity)
                end
                KernelAbstractions.synchronize(backend)
            end
        end
        wait.(exchangers[dim])
    end

    KernelAbstractions.synchronize(backend)

    # for dim in eachindex(neighbors)
    #     setdone!.(exchangers[dim])
    # end

    sleep(2me)
    @info "me == $me"
    display(A)

    finalize_distributed(; finalize_MPI=true)
    return
end

backend = CUDABackend()
T::DataType = Int
dims = (0, 0, 1)

main(backend, T, dims)