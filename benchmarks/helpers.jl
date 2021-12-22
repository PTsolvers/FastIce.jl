using DelimitedFiles, Plots, Interpolations

"""
    read_data(dat_file::String; resol::Int=128, visu=false)

Reads bedrock and ice elevation data for given x-coords from a text file and returns them interpolated on a finer grid.

The `dat_file` is expected to have 3 space-limited columns as `[x-coord  z-bedrock  z-surface]`
"""
function read_data(dat_file::String; resol::Int=128, visu=false)

    data = readdlm(dat_file, Float64)

    xc_d   = data[:,1]
    bed_d  = data[:,2]
    surf_d = data[:,3]

    xc    = LinRange(xc_d[1], xc_d[end], resol)
    itp1  = interpolate( (xc_d,), bed_d[:,1],  Gridded(Linear()))
    itp2  = interpolate( (xc_d,), surf_d[:,1], Gridded(Linear()))
    bed   = itp1.(xc)
    surf  = itp2.(xc)
    
    @assert length(xc) == size(bed)[1] == size(surf)[1]
    println("Interpolating original data (nx=$(size(bed_d)[1])) on nx=$(size(bed)[1]) grid.")

    if visu
        plot(xc, bed, label="bedrock", linewidth=3)
        display(plot!(xc, surf, label="surface", linewidth=3))
    end
    return xc, bed, surf
end

