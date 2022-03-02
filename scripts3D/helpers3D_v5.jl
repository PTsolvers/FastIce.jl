using Interpolations

"Axis-aligned bounding box"
struct AABB{T<:Real}
    xmin::T; xmax::T
    ymin::T; ymax::T
    zmin::T; zmax::T
end


"Construct AABB from coordinates"
AABB(xs,ys,zs) = AABB(extrema(xs)...,extrema(ys)...,extrema(zs)...)


"AABB extents"
function extents(box::AABB)
    return box.xmax-box.xmin, box.ymax-box.ymin, box.zmax-box.zmin
end


"AABB center"
function center(box::AABB{T}) where T
    half = convert(T,0.5)
    return half*(box.xmin+box.xmax), half*(box.ymin+box.ymax), half*(box.zmin+box.zmax)
end


"Dilate AABB by extending its limits around the center by certain fraction in each dimension"
function dilate(box::AABB, fractions)
    Δx,Δy,Δz = extents(box).*fractions
    return AABB(box.xmin-Δx, box.xmax+Δx, box.ymin-Δy, box.ymax+Δy, box.zmin-Δz, box.zmax+Δz)
end


"Create AABB enclosing both box1 and box2"
function union(box1::AABB, box2::AABB)
    return AABB(min(box1.xmin,box2.xmin),max(box1.xmax,box2.xmax),
                min(box1.ymin,box2.ymin),max(box1.ymax,box2.ymax),
                min(box1.zmin,box2.zmin),max(box1.zmax,box2.zmax))
end


"Create uniform grid of values"
function create_grid(box::AABB,size)
    return LinRange(box.xmin,box.xmax,size[1]),
           LinRange(box.ymin,box.ymax,size[2]),
           LinRange(box.zmin,box.zmax,size[3])
end


"Abstract type representing bedrock and ice elevation"
abstract type AbstractElevation{T<:Real} end

rotated_domain(dem::AbstractElevation) = domain(dem)
rotation(dem::AbstractElevation)       = [1. 0. 0.; 0. 1. 0.; 0. 0. 1.]


"Elevation data on grid"
struct DataElevation{T, M<:AbstractMatrix{T}} <: AbstractElevation{T}
    x::M; y::M; z_bed::M; z_surf::M
    rotation::M
    domain::AABB{T}
    rotated_domain::AABB{T}
end


function DataElevation(x,y,z_bed,z_surf,R)
    # get non-rotated domain
    domain = AABB(extrema(x)...,extrema(y)...,minimum(min.(z_bed,z_surf)),maximum(max.(z_bed,z_surf)))
    # rotate bed and surface
    bed_extents  = AABB(rotate_minmax(x, y, z_bed , R)...)
    surf_extents = AABB(rotate_minmax(x, y, z_surf, R)...)
    # get rotated domain
    rotated_domain = union(bed_extents, surf_extents)
    return DataElevation(x,y,z_bed,z_surf,R,domain,rotated_domain)
end


domain(dem::DataElevation)         = dem.domain
rotated_domain(dem::DataElevation) = dem.rotated_domain
rotation(dem::DataElevation)       = dem.rotation


"Get elevation data at specified coordinates"
function evaluate(dem::DataElevation, x::AbstractVector, y::AbstractVector)
    x1d, y1d = dem.x[:,1], dem.y[1,:]
    itp_bed  = interpolate( (x1d,y1d), dem.z_bed , Gridded(Linear()) )
    itp_surf = interpolate( (x1d,y1d), dem.z_surf, Gridded(Linear()) )
    return [itp_bed(_x,_y) for _x in x, _y in y], [itp_surf(_x,_y) for _x in x, _y in y]
end


"Load elevation data from HDF5 file."
function load_elevation(path::AbstractString)
    fid    = h5open(path, "r")
    x      = read(fid,"glacier/x")
    y      = read(fid,"glacier/y")
    z_bed  = read(fid,"glacier/z_bed")
    z_surf = read(fid,"glacier/z_surf")
    R      = read(fid,"glacier/R")
    close(fid)
    return DataElevation(x,y,z_bed,z_surf,R)
end


"Synthetic elevation data on grid."
struct SyntheticElevation{T, B, S} <: AbstractElevation{T}
    z_bed::B; z_surf::S
    domain::AABB{T}
end

domain(dem::SyntheticElevation) = dem.domain

"Get synthetic elevation data at specified coordinates."
function evaluate(dem::SyntheticElevation, x::AbstractVector, y::AbstractVector)
    return [dem.z_bed(_x,_y) for _x in x, _y in y], [dem.z_surf(_x,_y) for _x in x, _y in y]
end


# generate_z_surf(x,y,gl) = (gl*gl - (x+0.2*gl)*(x+0.2*gl)*0.85^2 - (y*gl)*(y*gl)*0.7^2)
generate_z_surf(x,y,gl) = gl^2 - ((x+0.2*gl)*0.85)^2 - ((y*gl)*0.5)^2
generate_z_bed(x,y,lx,ly,amp,ω,tanβ,el) = amp*sin(ω*x/lx)*sin(ω*y/ly) + tanβ*x + el + y^2/ly


"""
    generate_elevation(lx,ly,zminmax,amp,ω,tanβ,el,gl)

Generate synthetic elevation data for `lx`, `ly` and `zminmax=(zmin,zmax)` domain.
"""
function generate_elevation(lx,ly,zminmax,amp,ω,tanβ,el,gl)
    domain = AABB(-lx/2, lx/2, -ly/2, ly/2, zminmax[1], zminmax[2])
    z_bed  = (x,y) -> generate_z_bed(x,y,lx,ly,amp,ω,tanβ,el)
    z_surf = (x,y) -> generate_z_surf(x,y,gl)
    return SyntheticElevation(z_bed,z_surf,domain)
end


"Round the number of grid points that is optimal for GPUs."
function gpu_res(resol, t)
    resol = resol > t ? resol : t
    shift = resol % t
    return (shift < t/2 ? Int(resol - shift) : Int(resol + t - shift))
end


"Rotate field `X`, `Y`, `Z` with rotation matrix `R`."
function rotate(X, Y, Z, R)
    xrot = R[1,1].*X .+ R[1,2].*Y .+ R[1,3].*Z
    yrot = R[2,1].*X .+ R[2,2].*Y .+ R[2,3].*Z
    zrot = R[3,1].*X .+ R[3,2].*Y .+ R[3,3].*Z
    return xrot, yrot, zrot
end


"Rotate field `X`, `Y`, `Z` with rotation matrix `R` and return extents."
rotate_minmax(X, Y, Z, R) = extrema.(rotate(X, Y, Z, R))
