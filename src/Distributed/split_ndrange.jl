@inline subrange(nr,bw,I,::Val{1}) = 1:bw[I]
@inline subrange(nr,bw,I,::Val{2}) = (size(nr,I)-bw[I]+1):size(nr,I)
@inline subrange(nr,bw,I,::Val{3}) = (bw[I]+1):(size(nr,I)-bw[I])

@inline split_ndrange(ndrange,ndwidth) = split_ndrange(CartesianIndices(ndrange),ndwidth)

function split_ndrange(ndrange::CartesianIndices{N},ndwidth::NTuple{N,<:Integer}) where N
    @assert all(size(ndrange) .> ndwidth.*2)
    @inline ndsubrange(I,::Val{J}) where J = ntuple(Val(N)) do idim
        if idim < I
            1:size(ndrange,idim)
        elseif idim == I
            subrange(ndrange,ndwidth,idim,Val(J))
        else
            subrange(ndrange,ndwidth,idim,Val(3))
        end
    end
    ndinner = ntuple(idim -> subrange(ndrange,ndwidth,idim,Val(3)), Val(N))
    return ntuple(Val(2N+1)) do i
        if i == 2N+1
            ndrange[ndinner...]
        else
            idim,idir = divrem(i-1,2) .+ 1
            ndrange[ndsubrange(idim,Val(idir))...]
        end
    end
end