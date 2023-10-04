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
    # @show device = CUDA.device!(dev_id)
    # @show device = AMDGPU.device_id!(dev_id + 1)
    @show device = AMDGPU.device_id!(dev_id*2 + 1)
    return (dims, comm, me, neighbors, coords, device)
end

function finalize_distributed(; finalize_MPI=true)
    finalize_MPI && MPI.Finalize()
    return
end

# exchanger
mutable struct Exchanger
    @atomic done::Bool
    channel::Channel
    bottom::Base.Event
    @atomic err
    task::Task

    function Exchanger(backend::Backend, device)
        channel = Channel()
        bottom  = Base.Event(true)

        this = new(false, channel, bottom, nothing)

        recv_buf = nothing
        send_buf = nothing

        this.task = Threads.@spawn begin
            # CUDA.device!(device)
            AMDGPU.device!(device)
            KernelAbstractions.priority!(backend, :high)
            try
                while !(@atomic this.done)
                    f, comm, rank, halo, border = take!(channel)

                    has_neighbor = rank != MPI.PROC_NULL
                    compute_bc   = !has_neighbor

                    if isnothing(recv_buf)
                        recv_buf = similar(halo)
                        send_buf = similar(border)
                    end
                    if has_neighbor
                        recv = MPI.Irecv!(recv_buf, comm; source=rank)
                    end
                    f(compute_bc)
                    if has_neighbor
                        copyto!(send_buf, border)
                        AMDGPU.synchronize(blocking=false) #KernelAbstractions.synchronize(backend)
                        send = MPI.Isend(send_buf, comm; dest=rank)
                        flag = false
                        while true
                            test_recv = MPI.Test(recv)
                            test_send = MPI.Test(send)
                            if test_recv && !flag
                                copyto!(halo, recv_buf)
                                flag = true
                            end
                            if test_recv && test_send break end
                            yield()
                        end
                    end
                    AMDGPU.synchronize(blocking=false) #KernelAbstractions.synchronize(backend)
                    notify(bottom)
                end
            catch err
                @show err
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

function start_exchange(f, exc::Exchanger, comm, rank, halo, border)
    if !(@atomic exc.done)
        put!(exc.channel, (f, comm, rank, halo, border))
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
