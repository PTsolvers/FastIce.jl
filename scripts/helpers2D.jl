using Interpolations

"Axis-aligned bounding box"
struct AABB{T<:Real}
    xmin::T; xmax::T
    ymin::T; ymax::T
end


"Construct AABB from coordinates"
AABB(xs,ys) = AABB(extrema(xs)...,extrema(ys)...)


"AABB extents"
function extents(box::AABB)
    return box.xmax-box.xmin, box.ymax-box.ymin
end


"AABB center"
function center(box::AABB{T}) where T
    half = convert(T,0.5)
    return half*(box.xmin+box.xmax), half*(box.ymin+box.ymax)
end


"Dilate AABB by extending its limits around the center by certain fraction in each dimension"
function dilate(box::AABB, fractions)
    Δx,Δy = extents(box).*fractions
    return AABB(box.xmin-Δx, box.xmax+Δx, box.ymin-Δy, box.ymax+Δy)
end


"Create AABB enclosing both box1 and box2"
function union(box1::AABB, box2::AABB)
    return AABB(min(box1.xmin,box2.xmin),max(box1.xmax,box2.xmax),
                min(box1.ymin,box2.ymin),max(box1.ymax,box2.ymax))
end


"Create uniform grid of values"
function create_grid(box::AABB,size)
    return LinRange(box.xmin,box.xmax,size[1]),
           LinRange(box.ymin,box.ymax,size[2])
end


"Abstract type representing bedrock and ice elevation"
abstract type AbstractElevation{T<:Real} end

rotated_domain(dem::AbstractElevation) = domain(dem)
rotation(dem::AbstractElevation)       = [1. 0.; 0. 1.]


"Elevation data on grid"
struct DataElevation{T, M<:AbstractVector{T}} <: AbstractElevation{T}
    x::M; y_bed::M; y_surf::M
    rotation::M
    domain::AABB{T}
    rotated_domain::AABB{T}
end


function DataElevation(x,y_bed,y_surf,R)
    # get non-rotated domain
    domain = AABB(extrema(x)...,minimum(min.(y_bed,y_surf)),maximum(max.(y_bed,y_surf)))
    # rotate bed and surface
    bed_extents  = AABB(rotate_minmax(x, y_bed , R)...)
    surf_extents = AABB(rotate_minmax(x, y_surf, R)...)
    # get rotated domain
    rotated_domain = union(bed_extents, surf_extents)
    return DataElevation(x,y_bed,y_surf,R,domain,rotated_domain)
end


domain(dem::DataElevation)         = dem.domain
rotated_domain(dem::DataElevation) = dem.rotated_domain
rotation(dem::DataElevation)       = dem.rotation


"Get elevation data at specified coordinates"
function evaluate(dem::DataElevation, x::AbstractVector)
    x1d = dem.x
    itp_bed  = interpolate( x1d, dem.y_bed , Gridded(Linear()) )
    itp_surf = interpolate( x1d, dem.y_surf, Gridded(Linear()) )
    return [itp_bed(_x) for _x in x], [itp_surf(_x) for _x in x]
end


"Load elevation data from HDF5 file."
function load_elevation(path::AbstractString)
    fid    = h5open(path, "r")
    x      = read(fid,"glacier/x")
    y_bed  = read(fid,"glacier/y_bed")
    y_surf = read(fid,"glacier/y_surf")
    R      = read(fid,"glacier/R")
    close(fid)
    return DataElevation(x,y,y_bed,y_surf,R)
end


"Synthetic elevation data on grid."
struct SyntheticElevation{T, B, S} <: AbstractElevation{T}
    y_bed::B; y_surf::S
    domain::AABB{T}
end

domain(dem::SyntheticElevation) = dem.domain

"Get synthetic elevation data at specified coordinates."
function evaluate(dem::SyntheticElevation, x::AbstractVector)
    return [dem.y_bed(_x) for _x in x], [dem.y_surf(_x) for _x in x]
end


generate_y_surf(x,gl) = (gl*gl - (x+0.1*gl)*(x+0.1*gl))
generate_y_bed(x,lx,amp,ω,tanβ,el) = amp*sin(ω*x/lx) + tanβ*x + el


"""
    generate_elevation(lx,ly,zminmax,amp,ω,tanβ,el,gl)

Generate synthetic elevation data for `lx`, `ly` and `zminmax=(zmin,zmax)` domain.
"""
function generate_elevation(lx,yminmax,amp,ω,tanβ,el,gl)
    domain = AABB(-lx/2, lx/2, yminmax[1], yminmax[2])
    y_bed  = x -> generate_y_bed(x,lx,amp,ω,tanβ,el)
    y_surf = x -> generate_y_surf(x,gl)
    return SyntheticElevation(y_bed,y_surf,domain)
end


"Round the number of grid points that is optimal for GPUs."
function gpu_res(resol, t)
    resol = resol > t ? resol : t
    shift = resol % t
    return (shift < t/2 ? Int(resol - shift) : Int(resol + t - shift))
end


"Rotate field `X`, `Y`, `Z` with rotation matrix `R`."
function rotate(X, Y, R)
    xrot = R[1,1].*X .+ R[1,2].*Y
    yrot = R[2,1].*X .+ R[2,2].*Y
    return xrot, yrot
end


"Rotate field `X`, `Y`, `Z` with rotation matrix `R` and return extents."
rotate_minmax(X, Y, R) = extrema.(rotate(X, Y, R))
