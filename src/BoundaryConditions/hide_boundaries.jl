struct HideBoundaries{N}
    pipelines::NTuple{N,Tuple{Pipeline,Pipeline}}
    function HideBoundaries{N}(arch::Architecture) where {N}
        pre() = set_device_and_priority!(arch, :high)
        pipelines = ntuple(Val(N)) do _
            return ntuple(_ -> Pipeline(; pre), Val(2))
        end
        return new{N}(pipelines)
    end
end

function hide(fun::F, hb::HideBoundaries{N}, arch::Architecture, grid::CartesianGrid{N}, boundary_conditions, worksize;
              outer_width=nothing) where {F,N}
    inner_range, outer_ranges = split_ndrange(worksize, outer_width)
    # execute inner range in a main Task with a normal priority
    fun(inner_range)
    for dim in N:-1:1
        ntuple(Val(2)) do side
            pipe  = hb.pipelines[dim][side]
            range = outer_ranges[dim][side]
            batch = boundary_conditions[dim][side]
            # execute outer range and boundary conditions asynchronously
            put!(pipe) do
                fun(range)
                apply_boundary_conditions!(Val(side), Val(dim), arch, grid, batch)
                synchronize(backend(arch))
            end
        end
        wait.(hb.pipelines[dim]) # synchronize spatial dimension
    end
    return
end
