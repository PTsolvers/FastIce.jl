# MPI
function init_distributed(dims::Tuple=(0, 0, 0); init_MPI=true)
    init_MPI && MPI.Init()
    nprocs = MPI.Comm_size(MPI.COMM_WORLD)
    dims = Tuple(MPI.Dims_create(nprocs, dims))
    # create MPI communicator
    comm = MPI.Cart_create(MPI.COMM_WORLD, dims)
    me = MPI.Comm_rank(comm)
    neighbors = ntuple(Val(length(dims))) do idim
        MPI.Cart_shift(comm, idim-1, 1)
    end
    coords = Tuple(MPI.Cart_coords(comm))
    # create communicator for the node and select device
    comm_node = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, me)
    dev_id = MPI.Comm_rank(comm_node)
    @show CUDA.device!(dev_id)
    # @show AMDGPU.default_device_id!(dev_id + 1) # DEBUG: why default ???
    # @show AMDGPU.device_id!(dev_id + 1)
    return (dims, comm, me, neighbors, coords)
end

function finalize_distributed(; finalize_MPI=true)
    finalize_MPI && MPI.Finalize()
    return
end

# TODO: Implement in MPI.jl
function cooperative_test!(req)
    done = false
    while !done
        done, _ = MPI.Test(req, MPI.Status)
        yield()
    end
end

# exchanger
mutable struct Exchanger
    @atomic done::Bool
    top::Base.Event
    bottom::Base.Event
    @atomic err
    task::Task

    function Exchanger(f::F, backend::Backend, rank, halo, border) where F
        top    = Base.Event(#=autoreset=# true)
        bottom = Base.Event(#=autoreset=# true)

        send_buf = similar(border)
        recv_buf = similar(halo)
        this = new(false, top, bottom, nothing)

        this.task = Threads.@spawn begin
            KernelAbstractions.priority!(backend, :high)
            try
                while !(@atomic this.done)
                    wait(top)
                    NVTX.@mark "after wait(top)"
                    if rank != -1
                        recv = MPI.Irecv!(recv_buf, comm; source=rank)
                    end
                    f(rank == -1)
                    if rank != -1
                        copyto!(send_buf, border)
                        send = MPI.Isend(send_buf, comm; dest=rank)
                        cooperative_test!(recv)
                        copyto!(halo, recv_buf)
                        cooperative_test!(send)
                    end
                    notify(bottom)
                    NVTX.@mark "after notify(bottom)"
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

get_recv_view(::Val{1}, ::Val{D}, A) where D = view(A, ntuple(I -> I == D ? 1          : Colon(), Val(ndims(A)))...)
get_recv_view(::Val{2}, ::Val{D}, A) where D = view(A, ntuple(I -> I == D ? size(A, D) : Colon(), Val(ndims(A)))...)

get_send_view(::Val{1}, ::Val{D}, A) where D = view(A, ntuple(I -> I == D ? 2              : Colon(), Val(ndims(A)))...)
get_send_view(::Val{2}, ::Val{D}, A) where D = view(A, ntuple(I -> I == D ? size(A, D) - 1 : Colon(), Val(ndims(A)))...)
