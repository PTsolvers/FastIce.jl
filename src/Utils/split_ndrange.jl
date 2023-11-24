@inline outer_subrange(nr, bw, D, ::Val{1}) = 1:bw[D]
@inline outer_subrange(nr, bw, D, ::Val{2}) = (size(nr, D)-bw[D]+1):size(nr, D)
@inline inner_subrange(nr, bw, D) = (bw[D]+1):(size(nr, D)-bw[D])

function ndsubrange(ndrange::CartesianIndices{N}, ndwidth, I, ::Val{S}) where {N,S}
    return ntuple(Val(N)) do D
        if D < I
            1:size(ndrange, D)
        elseif D == I
            outer_subrange(ndrange, ndwidth, D, Val(S))
        else
            inner_subrange(ndrange, ndwidth, D)
        end
    end
end

@inline split_ndrange(ndrange, ndwidth) = split_ndrange(CartesianIndices(ndrange), ndwidth)

function split_ndrange(ndrange::CartesianIndices{N}, ndwidth::NTuple{N,<:Integer}) where {N}
    @assert all(size(ndrange) .> ndwidth .* 2)
    ndinner = ntuple(D -> inner_subrange(ndrange, ndwidth, D), Val(N))
    inner_range = ndrange[ndinner...]
    outer_ranges = ntuple(D -> (ndrange[ndsubrange(ndrange, ndwidth, D, Val(1))...],
                                ndrange[ndsubrange(ndrange, ndwidth, D, Val(2))...]), Val(N))
    return inner_range, outer_ranges
end
