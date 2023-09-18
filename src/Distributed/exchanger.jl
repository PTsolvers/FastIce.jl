mutable struct Exchanger
    @atomic done::Bool
    ch::Channel
    bottom::Base.Event
    task::Task
    @atomic err

    function Exchanger(f::F, arch::AbstractArchitecture, comm, rank, halo, border) where F
        top    = Base.Event(true)
        bottom = Base.Event(true)

        send_buf = similar(border)
        recv_buf = similar(halo)
        this = new(false, top, bottom, nothing)

        has_neighbor = rank != MPI.PROC_NULL
        compute_bc   = !has_neighbor

        this.task = Threads.@spawn begin
            set_device!(device(arch))
            KernelAbstractions.priority!(backend(arch), :high)
            try
                while !(@atomic this.done)
                    wait(top)
                    if has_neighbor
                        recv = MPI.Irecv!(recv_buf, comm; source=rank)
                    end
                    f(compute_bc)
                    if has_neighbor
                        copyto!(send_buf, border)
                        send = MPI.Isend(send_buf, comm; dest=rank)
                        cooperative_test!(recv)
                        copyto!(halo, recv_buf)
                        cooperative_test!(send)
                    end
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
