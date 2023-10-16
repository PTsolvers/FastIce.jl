struct DistributedBoundaryConditions{F,B}
    fields::F
    exchange_infos::B

    function DistributedBoundaryConditions(::Val{S}, ::Val{D}, fields::NTuple{N}) where {S,D,N}
        exchange_infos = ntuple(Val(N)) do idx
            send_view = get_send_view(Val(S), Val(D), fields[idx])
            recv_view = get_recv_view(Val(S), Val(D), fields[idx])
            send_buffer = similar(parent(send_view), eltype(send_view), size(send_view))
            recv_buffer = similar(parent(recv_view), eltype(recv_view), size(recv_view))
            ExchangeInfo(send_buffer, recv_buffer)
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

function apply_boundary_conditions!(::Val{S}, ::Val{D}, arch::Architecture, grid::CartesianGrid,
                                    bc::DistributedBoundaryConditions; async=true) where {S,D}
    comm = cartesian_communicator(details(arch))
    nbrank = neighbor(details(arch), D, S)

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

    recv_ready = BitVector(false for _ in eachindex(bc.exchange_infos))
    send_ready = BitVector(false for _ in eachindex(bc.exchange_infos))

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

_overlap(::Vertex) = 1
_overlap(::Center) = 0

get_recv_view(side::Val{S}, dim::Val{D}, f::Field) where {S,D} = get_recv_view(side, dim, parent(f), halo(f, D, S))
get_send_view(side::Val{S}, dim::Val{D}, f::Field) where {S,D} = get_send_view(side, dim, parent(f), halo(f, D, S), _overlap(location(f, dim)))

function get_recv_view(::Val{1}, ::Val{D}, array::AbstractArray, halo_width::Integer) where {D}
    recv_range = Base.OneTo(halo_width)
    indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_recv_view(::Val{2}, ::Val{D}, array::AbstractArray, halo_width::Integer) where {D}
    recv_range = (size(array, D)-halo_width+1):size(array, D)
    indices = ntuple(I -> I == D ? recv_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_send_view(::Val{1}, ::Val{D}, array::AbstractArray, halo_width::Integer, overlap::Integer) where {D}
    send_range = (overlap+halo_width+1):(overlap+2halo_width)
    indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end

function get_send_view(::Val{2}, ::Val{D}, array::AbstractArray, halo_width::Integer, overlap::Integer) where {D}
    send_range = (size(array, D)-overlap-2halo_width+1):(size(array, D)-overlap-halo_width)
    indices = ntuple(I -> I == D ? send_range : Colon(), Val(ndims(array)))
    return view(array, indices...)
end
