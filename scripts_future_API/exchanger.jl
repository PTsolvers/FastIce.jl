using KernelAbstractions
using MPI

include("mpi_utils.jl")

mutable struct Exchanger
    @atomic done::Bool
    top::Base.Event
    bottom::Base.Event
    @atomic err
    task::Task

    function Exchanger(f::F, backend::Backend) where F
        top = Base.Event(#=autoreset=# true)
        bottom = Base.Event(#=autoreset=# true)
        this = new(false, top, bottom, nothing)

        this.task = Threads.@spawn begin
            KernelAbstractions.priority!(backend, :high)
            try
                while !(@atomic this.done)
                    wait(top)
                    f()
                    notify(bottom)
                end
            catch err
                @atomic this.done = true
                @atomic this.err = err
            end
        end
        errormonitor(this.task)
        return this
    end
end

setdone!(exc::Exchanger) = @atomic exc.done = true

Base.isdone(exc::Exchanger) = @atomic exc.done
function Base.notify(exc::Exchanger)
    if !(@atomic exc.done)
        notify(exc.top)
    else
        error("notify: Exchanger is not running")
    end
end
function Base.wait(exc::Exchanger)
    if !(@atomic exc.done)
        wait(exc.bottom)
    else
        error("wait: Exchanger is not running")
    end
end

# TODO: Implement in MPI.jl
function cooperative_test!(req)
    done = false
    while !done
        done, _ = MPI.Test(req, MPI.Status)
        yield()
    end
end

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

backend = CPU()
T::DataType = Int
dims = (0, 0, 0)

# function main(backend = CPU(), T::DataType = Float64, dims = (0, 0, 0))
MPI.Init()

nprocs = MPI.Comm_size(MPI.COMM_WORLD)

dims = Tuple(MPI.Dims_create(nprocs, dims))

# create MPI communicator
comm      = MPI.Cart_create(MPI.COMM_WORLD, dims)
me        = MPI.Comm_rank(comm)
neighbors = ntuple(Val(length(dims))) do idim
    MPI.Cart_shift(comm, idim-1, 1)
end
coords = Tuple(MPI.Cart_coords(comm))
# create communicator for the node and select device
comm_node = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, me)
pid       = MPI.Comm_rank(comm_node)
## TODO: set device with pid
# CUDA.device!(pid)

nx, ny, nz = 6, 6, 6
bx, by, bz = 2, 2, 2
A = KernelAbstractions.allocate(backend, T, nx, ny, nz)
fill!(A, -1)

ranges = split_ndrange(A, (bx, by, bz))

get_recv_view(::Val{1}, ::Val{D}, A) where D = view(A, ntuple(I -> I == D ? 1          : Colon(), Val(ndims(A)))...)
get_recv_view(::Val{2}, ::Val{D}, A) where D = view(A, ntuple(I -> I == D ? size(A, D) : Colon(), Val(ndims(A)))...)

get_send_view(::Val{1}, ::Val{D}, A) where D = view(A, ntuple(I -> I == D ? 2              : Colon(), Val(ndims(A)))...)
get_send_view(::Val{2}, ::Val{D}, A) where D = view(A, ntuple(I -> I == D ? size(A, D) - 1 : Colon(), Val(ndims(A)))...)

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

MPI.Finalize()

#     return
# end

# main()