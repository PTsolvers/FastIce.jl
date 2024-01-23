"""
    gather!(arch::Architecture{DistributedMPI}, dst, src::Field; kwargs...)

Gather the interior of a field `src` into a global array `dst`.
"""
function gather!(arch::Architecture{DistributedMPI}, dst, src::Field; kwargs...)
    ImplicitGlobalGrid.gather!(interior(src), dst, cartesian_communicator(details(arch)); kwargs...)
end
