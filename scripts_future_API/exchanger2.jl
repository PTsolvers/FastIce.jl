using KernelAbstractions
using MPI
using CUDA

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
    dims, comm, me, neighbors, coords = init_distributed(dims; init_MPI=true)

    nx, ny, nz = 6, 6, 6
    b_width = (2, 2, 2)

    # init arrays
    A = KernelAbstractions.allocate(backend, T, nx, ny, nz)
    fill!(A, -1)

    ranges = split_ndrange(A, b_width)

    exchangers = ntuple(Val(length(neighbors))) do dim
        ntuple(2) do side
            Exchanger(backend) do
                rank = neighbors[dim][side]
                if rank != -1
                    recv_buf = get_recv_view(Val(side), Val(dim), A)
                    recv = MPI.Irecv!(recv_buf,comm;source=rank)
                end

                I = 2*(dim-1) + side

                do_work!(backend, 256)(A, me, first(ranges[I]); ndrange=size(ranges[I]))
                KernelAbstractions.synchronize(backend)

                if rank != -1
                    send_buf = get_send_view(Val(side), Val(dim), A)
                    send = MPI.Isend(send_buf,comm;dest=rank)
                    cooperative_test!(recv)
                    cooperative_test!(send)
                end
            end
        end
    end

    do_work!(backend, 256)(A, me, first(ranges[end]); ndrange=size(ranges[end]))

    for dim in reverse(eachindex(neighbors))
        notify.(exchangers[dim])
        wait.(exchangers[dim])
    end

    KernelAbstractions.synchronize(backend)

    # for dim in eachindex(neighbors)
    #     setdone!.(exchangers[dim])
    # end

    sleep(me)
    @info "me == $me"
    display(A)

    finalize_distributed(; finalize_MPI=true)
    return
end

backend = CUDABackend()
T::DataType = Int
dims = (0, 0, 1)

main(backend, T, dims)