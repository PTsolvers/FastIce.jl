struct HideBoundaries{N}
    workers::NTuple{N,Tuple{Worker,Worker}}
    outer_width::NTuple{N,Int}
    function HideBoundaries(arch::Architecture, outer_width::NTuple{N,Int}) where {N}
        setup() = set_device_and_priority!(arch, :high)
        workers = ntuple(D -> (Worker(; setup), Worker(; setup)), Val(N))
        return new{N}(workers, outer_width)
    end
end

function hide(fun::F, hb::HideBoundaries{N}, arch::Architecture, grid::CartesianGrid{N}, boundary_conditions, worksize) where {F,N}
    inner_range, outer_ranges = split_ndrange(worksize, hb.outer_width)
    # execute inner range in a parent Task with a normal priority
    fun(inner_range)
    for dim in N:-1:1
        ntuple(Val(2)) do side
            worker = hb.workers[dim][side]
            range = outer_ranges[dim][side]
            batch = boundary_conditions[dim][side]
            # execute outer range and boundary conditions asynchronously
            put!(worker) do
                fun(range)
                apply_boundary_conditions!(Val(side), Val(dim), arch, grid, batch)
                synchronize(backend(arch))
            end
        end
        wait.(hb.workers[dim]) # synchronize workers for spatial dimension
    end
    return
end
