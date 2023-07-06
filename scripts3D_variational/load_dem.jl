using FileIO

function filter_range(r,lims)
    istart = something(findfirst(v -> v>lims[1], r), length(r))
    iend   = something( findlast(v -> v<lims[2], r), 1)
    return istart:iend
end

function load_dem(path,(;xlims,ylims))
    x,y,bed,surface=load(path,"x","y","bed","surface")
    # shift limits to look for offsets
    ixs = filter_range(x,xlims .+ x[1])
    iys = filter_range(y,ylims .+ y[1])
    return (x = x[ixs], y = y[iys], bed = bed[ixs,iys], surface = surface[ixs,iys])
end