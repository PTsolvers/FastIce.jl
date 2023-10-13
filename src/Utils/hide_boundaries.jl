struct HideBoundaries{N,IR,OR}
    pipelines::NTuple{N,Tuple{Pipeline,Pipeline}}
    inner_range::IR
    outer_ranges::OR
    function HideBoundaries(arch::AbstractArchitecture, ndrange::CartesianIndices{N}, ndwidth::NTuple{N,Int}) where {N}
        pre() = set_device_and_priority!(arch, :high)
        pipelines = ntuple(Val(N)) do _
            return ntuple(_ -> Pipeline(; pre), Val(2))
        end
        ranges = split_ndrange(ndrange, ndwidth)
        inner_range = ranges[end]
        outer_ranges = ranges[1:end-1]
        return new{N,typeof(inner_range),typeof(outer_ranges)}(pipelines, inner_range, outer_ranges)
    end
end

function hide(fun::F, hb::HideBoundaries{N}, arch::AbstractArchitecture, grid, fields, boundary_conditions::NTuple{N,Tuple}) where {F,N}
    fun(hb.inner_range)
    for dim in N:-1:1
        ntuple(Val(2)) do side
            println("dim = $dim, side = $side")
            pipe  = hb.pipelines[dim][side]
            range = hb.outer_ranges[dim][side]
            bcs   = boundary_conditions[dim][side]
            # execute outer range and boundary conditions asynchronously
            put!(pipe) do
                fun(range)
                apply_boundary_conditions!(Val(dim), backend(arch), grid, fields, bcs)
                return Architectures.synchronize(arch)
            end
        end
        wait.(hb.pipelines[dim]) # synchronize spatial dimension
    end
    return
end
