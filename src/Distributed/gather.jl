function gather!(dst::Union{AbstractArray,Nothing}, src::AbstractArray, comm::MPI.Comm; root=0)
    dims, _, _ = MPI.Cart_get(comm)
    dims = Tuple(dims)
    if MPI.Comm_rank(comm) == root
        # make subtype for gather
        offset  = Tuple(0 for _ in 1:ndims(src))
        subtype = MPI.Types.create_subarray(size(dst), size(src), offset, MPI.Datatype(eltype(dst)))
        subtype = MPI.Types.create_resized(subtype, 0, size(src, 1) * Base.elsize(dst))
        MPI.Types.commit!(subtype)
        # make VBuffer for collective communication
        counts = fill(Cint(1), dims)
        displs = zeros(Cint, dims)
        d = zero(Cint)
        for j in 1:dims[2]
            for i in 1:dims[1]
                displs[i, j] = d
                d += 1
            end
            d += (size(src, 2) - 1) * dims[1]
        end
        # transpose displs as cartesian communicator is row-major
        recvbuf = MPI.VBuffer(dst, vec(counts), vec(displs'), subtype)
        MPI.Gatherv!(src, recvbuf, comm; root)
    else
        MPI.Gatherv!(src, nothing, comm; root)
    end
    return
end

function gather!(arch::Architecture{DistributedMPI}, dst, src::Field; kwargs...)
    gather!(dst, interior(src), cartesian_communicator(details(arch)); kwargs...)
end
