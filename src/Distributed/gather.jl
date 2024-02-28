"""
    gather!(dst::Union{AbstractArray{T,N},Nothing}, src::AbstractArray{T,N}, comm::MPI.Comm; root=0) where {T,N}

Gather local array `src` into a global array `dst`.
Size of the global array `size(dst)` should be equal to the product of the size of a local array `size(src)` and the dimensions of a Cartesian communicator `comm`.
The array will be gathered on the process with id `root` (`root=0` by default).
Note that the memory for a global array should be allocated only on the process with id `root`, on other processes `dst` can be set to `nothing`.
"""
function gather!(dst::Union{AbstractArray{T,N},Nothing}, src::AbstractArray{T,N}, comm::MPI.Comm; root=0) where {T,N}
    ImplicitGlobalGrid.gather!(src, dst, comm; root=root)
end

"""
    gather!(arch::Architecture{DistributedMPI}, dst, src::Field; kwargs...)

Gather the interior of a field `src` into a global array `dst`.
"""
function gather!(arch::Architecture{DistributedMPI}, dst, src::Field; kwargs...)
    gather!(dst, interior(src), cartesian_communicator(details(arch)); kwargs...)
end
