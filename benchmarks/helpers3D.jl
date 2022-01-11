using DelimitedFiles, Plots, Interpolations, MAT

"Heuristic number of threads per block in x-dim."
const tx = 32

"Heuristic number of threads per block in y-dim."
const ty = 8

"Heuristic number of threads per block in z-dim."
const tz = 8

"Input parameters structure"
mutable struct InputParams3D
    zbedv
    zbedc
    zsurfv
    zsurfc
    maskv
    maskc
    xv
    yv
    zv
    xc
    yc
    zc
    lx_lz
    ly_lz
    max∆z
    nx::Int
    ny::Int
    nz::Int
    αx
    αy
end


"Average in x and y direction"
av(A) = 0.25(A[1:end-1,1:end-1].+A[1:end-1,2:end].+A[2:end,1:end-1].+A[2:end,2:end])


"""
    gpu_res(resol, t, olen)

Round the number of grid points optimally for GPUs.
"""
function gpu_res(resol, t, olen)
    resol = resol > t ? resol : t
    shift = resol % t
    return (shift < t/2 ? Int(resol - shift) : Int(resol + t - shift)) - olen
end


"""
    interp(A, B, C, xv_d, yv_d, xv, yv)

Linear interpolation of fields `A`, `B`, and `C` on `(xv, yv)` grid.
"""
@views function interp(A, B, C, xv_d, yv_d, xv, yv)

    itp1 = interpolate( (xv_d,yv_d), A, Gridded(Linear()))
    itp2 = interpolate( (xv_d,yv_d), B, Gridded(Linear()))
    itp3 = interpolate( (xv_d,yv_d), C, Gridded(Linear()))
    
    A2   = [itp1(x,y) for x in xv, y in yv]
    B2   = [itp2(x,y) for x in xv, y in yv]
    C2   = [itp3(x,y) for x in xv, y in yv]
    
    @assert length(xv) == size(A2)[1] == size(A2)[1]
    @assert length(yv) == size(A2)[2] == size(A2)[2]
    return A2, B2, C2
end


"""
    lsq_fitmask, zavg, xv2, yv2)

Linear least-square fit of mean bedrock and surface data.
"""
@views function lsq_fit(mask, zavg, xv2, yv2)
    # remove masked points
    xv2_    = [xv2[i]  for i in eachindex(xv2)  if mask[i] != 0]
    yv2_    = [yv2[i]  for i in eachindex(yv2)  if mask[i] != 0]
    zavg_   = [zavg[i] for i in eachindex(zavg) if mask[i] != 0]
    # prepare least square (lsq)
    A       =  ones(length(xv2_[:]),3)
    B       = zeros(length(xv2_[:]),1)
    A[:,1] .= xv2_[:]
    A[:,2] .= yv2_[:]
    B      .= zavg_[:]
    # lsq solve x (x-slope, y-slope, origin)
    x       = (A'*A) \ (A'*B)
    # best fitting plane
    plane   = xv2*x[1] .+ yv2*x[2] .+ x[3];
    return plane, x
end


"""
    rotate(zbed, zsurf, mask, plane)

Rotate bedrock `zbed` and surface `zsurf` profiles. 
"""
function rotate(zbed, zsurf, mask, plane)
    zbedr  = (zbed  .- plane) .* mask
    zsurfr = (zsurf .- plane) .* mask
    zmin   = minimum(zbedr)
    zbedr  = zbedr  .- zmin
    zsurfr = zsurfr .- zmin
    return zbedr, zsurfr
end


"""
    load_data(dat_file::String, do_interp::Bool, resolx::Int, resoly::Int, olen::Int)

Load bedrock and ice surface elevation data and return the interpolated data.

# Arguments
- `dat_file::String`: input data file with 3 space-limited columns as `[x-coord  z-bedrock  z-surface]`
- `do_interp::Bool`: perform data interpolation
- `resolx::Int`: output x-resolution
- `resoly::Int`: output y-resolution
- `olen::Int`: overlength for arrays larger then `nx`, `ny` and `nz`
"""
@views function load_data(dat_file::String, do_interp::Bool, resolx::Int, resoly::Int, olen::Int)
    
    println("- Loading the data")
    vars   = matread(dat_file)
    bed_d  = get(vars,"zbed" ,1)
    surf_d = get(vars,"zsurf",1)
    mask_d = get(vars,"mask" ,1)
    xv_d   = get(vars,"xv"   ,1); xv_d = vec(Float64.(xv_d))
    yv_d   = get(vars,"yv"   ,1); yv_d = vec(Float64.(yv_d))

    resx   = gpu_res(resolx, tx, olen)
    resy   = gpu_res(resoly, ty, olen)
    
    xv     = LinRange(xv_d[1], xv_d[end], resx+1)
    yv     = LinRange(yv_d[1], yv_d[end], resy+1)
    
    if do_interp
        bed, surf, mask = interp(bed_d, surf_d, mask_d, xv_d, yv_d, xv, yv)
        println("- Interpolating original data (nxv, nyv = $(size(bed_d)[1]), $(size(bed_d)[2])) on nxv, nyv = $(size(bed)[1]), $(size(bed)[2]) grid")
    else
        bed, surf, mask = bed_d, surf_d, mask_d
        println("- Using original data (nxv, nyv = $(size(bed_d)[1]), $(size(bed_d)[2])) grid")
    end
    
    # avoid interp noise (avg in mask)
    mask[mask.<1.0] .= 0.0
    
    return bed, surf, mask, xv, yv
end


"""
    preprocess(dat_file::String; do_interp=true, resolx::Int=128, resoly::Int=128, do_rotate=true, fact_nz::Int=1, do_nondim=true, olen::Int=1)

Preprocess input data for iceflow model.

# Arguments
- `dat_file::String`: input data file with 3 space-limited columns as `[x-coord  z-bedrock  z-surface]`
- `do_interp=true`: perform data interpolation
- `resolx::Int=128`: output x-resolution
- `resoly::Int=128`: output y-resolution
- `do_rotate=true`: perform data rotation
- `fact_nz::Int=1`: grid-point increase in z-dim
- `do_nondim=true`: perform non-dimensionalisation
- `olen::Int=1`: overlength for arrays larger then `nx`, `ny` and `nz`
"""
@views function preprocess(dat_file::String; do_interp::Bool=true, resolx::Int=128, resoly::Int=128, do_rotate::Bool=true, fact_nz::Int=1, do_nondim::Bool=true, olen::Int=1)
    
    println("Starting preprocessing ... ")

    # load the data
    zbed, zsurf, mask, xv, yv = load_data(dat_file, do_interp, resolx, resoly, olen)

    xv2  = repeat(xv ,1,length(yv)) .* mask
    yv2  = repeat(yv',length(xv),1) .* mask
    zavg = 0.5(zsurf .+ zbed) .* mask

    plane, x = lsq_fit(mask, zavg, xv2, yv2)

    if do_rotate
        zbedr, zsurfr = rotate(zbed, zsurf, mask, plane)
    else
        zbedr, zsurfr = zbed .* mask, zsurf .* mask
    end

    zmin   = minimum(zbedr)
    zbedr  = zbedr  .- zmin
    zsurfr = zsurfr .- zmin
    max∆z  = maximum(zsurf .- zbed)
    lz     = maximum(zsurfr)
    lx, ly = maximum(xv) - minimum(xv), maximum(yv) - minimum(yv)
    xc, yc = 0.5(xv[1:end-1].+xv[2:end]), 0.5(yv[1:end-1].+yv[2:end])
    nx, ny = length(xc), length(yc)
    resz   = ceil(Int, lz/lx*nx)
    resz   = fact_nz * gpu_res(resz, tz, 0) - olen
    zv     = LinRange(0,lz,resz+1)
    zc     = 0.5(zv[1:end-1].+zv[2:end])
    nz     = length(zc)

    println("- Preprocessed data: nx=$nx, ny=$ny, nz=$nz (dx=$(round(lx/nx, sigdigits=4)), dy=$(round(ly/ny, sigdigits=4)), dz=$(round(lz/nz, sigdigits=4)))")
    
    maskc  = av(mask)
    maskc[maskc.<1.0] .= 0.0

    sc     = do_nondim ? 1.0/lz : 1.0    
    inputs = InputParams3D(zbedr*sc, av(zbedr)*sc, zsurfr*sc, av(zsurfr)*sc, mask, maskc, xv*sc, yv*sc, zv*sc, xc*sc, yc*sc, zc*sc, lx*sc, ly*sc, max∆z*sc, nx, ny, nz, x[1], x[2])
    
    println("... done.")
    return inputs, zavg, plane, x
end

# preprocessing
inputs, zavg, plane, x = preprocess("../data/arolla3D.mat"; resolx=512, resoly=512, fact_nz=6)


bed_v   = copy(inputs.zbedv);  bed_v[inputs.maskv.==0] .=NaN
surf_v  = copy(inputs.zsurfv); surf_v[inputs.maskv.==0].=NaN
thick_v = copy(inputs.zsurfv.-inputs.zbedv); thick_v[inputs.maskv.==0].=NaN

xvp, yvp = inputs.xv, inputs.yv
opts = ( aspect_ratio=1, xlims=(xvp[1],xvp[end]), ylims=(yvp[1],yvp[end]) )
p1 = heatmap(xvp, yvp, bed_v'; opts... )
p2 = heatmap(xvp, yvp, surf_v'; opts... )
p3 = heatmap(xvp, yvp, thick_v'; opts... )

display(plot(p1, p2, p3, layout = (3,1), size = (600,800), dpi = 200))
