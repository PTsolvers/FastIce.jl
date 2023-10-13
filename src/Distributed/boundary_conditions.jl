struct DistributedBoundaryConditions{F,B}
    fields::F
    exchange_infos::B

    function DistributedBoundaryConditions(::Val{S}, ::Val{D}, fields::NTuple{N}) where {S,D,N}
        exchange_infos = ntuple(Val(N)) do idx
            send_view = get_send_view(Val(S), Val(D), fields[idx])
            recv_view = get_recv_view(Val(S), Val(D), fields[idx])
            ExchangeInfo(similar(send_view), similar(recv_view))
        end
        return new{typeof(fields),typeof(exchange_infos)}(fields, exchange_infos)
    end
end

mutable struct ExchangeInfo{SB,RB}
    send_buffer::SB
    recv_buffer::RB
    send_request::MPI.Request
    recv_request::MPI.Request
end

ExchangeInfo(send_buf, recv_buf) = ExchangeInfo(send_buf, recv_buf, MPI.REQUEST_NULL, MPI.REQUEST_NULL)

function apply_boundary_conditions!(::Val{S}, ::Val{D}, arch::DistributedArchitecture, grid::CartesianGrid,
                                    bc::DistributedBoundaryConditions; async=true) where {S,D}
    comm = cartesian_communicator(arch.topology)
    nbrank = neighbor(arch.topology, D, S)

    # initiate non-blocking MPI recieve and device-to-device copy to the send buffer
    for idx in eachindex(bc.fields)
        info = bc.exchange_infos[idx]
        info.recv_request = MPI.Irecv!(info.recv_buffer, comm; source=nbrank)
        send_view = get_send_view(Val(S), Val(D), bc.fields[idx])
        copyto!(info.send_buffer, send_view)
    end
    Architectures.synchronize(arch)

    # initiate non-blocking MPI send
    for idx in eachindex(bc.fields)
        info = bc.exchange_infos[idx]
        info.send_request = MPI.Isend(info.send_buffer, comm; dest=nbrank)
    end

    recv_ready = BitVector(false for _ in eachindex(recv_requests))
    send_ready = BitVector(false for _ in eachindex(send_requests))

    # test send and receive requests, initiating device-to-device copy
    # to the receive buffer if the receive is complete
    while !(all(recv_ready) && all(send_ready))
        for idx in eachindex(bc.fields)
            info = bc.exchange_infos[idx]
            if MPI.Test(info.recv_request) && !recv_ready[idx]
                recv_view = get_recv_view(Val(S), Val(D), bc.fields[idx])
                copyto!(recv_view, info.recv_buffer)
                recv_ready[idx] = true
            end
            send_ready[idx] = MPI.Test(info.send_request)
        end
        yield()
    end
    async || Architectures.synchronize(arch)

    return
end

get_recv_view(::Val{1}, ::Val{D}, A) where {D} = view(A, ntuple(I -> I == D ? 0 : Colon(), Val(ndims(A)))...)
get_recv_view(::Val{2}, ::Val{D}, A) where {D} = view(A, ntuple(I -> I == D ? size(A, D) + 1 : Colon(), Val(ndims(A)))...)

get_send_view(::Val{1}, ::Val{D}, A) where {D} = view(A, ntuple(I -> I == D ? 1 : Colon(), Val(ndims(A)))...)
get_send_view(::Val{2}, ::Val{D}, A) where {D} = view(A, ntuple(I -> I == D ? size(A, D) : Colon(), Val(ndims(A)))...)
