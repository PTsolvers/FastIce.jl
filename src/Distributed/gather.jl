function gather!(dst::Union{AbstractArray{T,N},Nothing}, src::AbstractArray{T,N}, comm::MPI.Comm; root=0) where {T,N}
    dims, _, _ = MPI.Cart_get(comm)
    dims = Tuple(dims)
    if MPI.Comm_rank(comm) == root
        # make subtype for gather
        offset  = Tuple(0 for _ in 1:N)
        subtype = MPI.Types.create_subarray(size(dst), size(src), offset, MPI.Datatype(eltype(dst)))
        subtype = MPI.Types.create_resized(subtype, 0, size(src, 1) * Base.elsize(dst))
        MPI.Types.commit!(subtype)
        # make VBuffer for collective communication
        counts  = fill(Cint(1), reverse(dims)) # gather one subarray from each MPI rank
        displs  = zeros(Cint, reverse(dims))   # reverse dims since MPI Cart comm is row-major
        csizes  = cumprod(size(src)[2:end] .* dims[1:end-1])
        strides = (1, csizes...)
        for I in CartesianIndices(displs)
            offset = reverse(Tuple(I - oneunit(I)))
            displs[I] = sum(offset .* strides)
        end
        recvbuf = MPI.VBuffer(dst, vec(counts), vec(displs), subtype)
        MPI.Gatherv!(src, recvbuf, comm; root)
    else
        MPI.Gatherv!(src, nothing, comm; root)
    end
    return
end

function gather!(arch::Architecture{DistributedMPI}, dst, src::Field; kwargs...)
    gather!(dst, interior(src), cartesian_communicator(details(arch)); kwargs...)
end
