module Geometry

# export AbstractElevation, AABB, DataElevation, SyntheticElevation
# export domain, rotated_domain, rotation, generate_elevation, extents, center, dilate

abstract type AbstractElevation{T<:Real} end

struct AABB{T<:Union{Real}}
    xmin::T
    xmax::T
    ymin::T
    ymax::T
    zmin::T
    zmax::T
end

struct DataElevation{T,V<:AbstractArray{T},R<:AbstractRange{T},M<:AbstractMatrix{T}} <: AbstractElevation{T}
    x::R
    y::R
    offsets::V
    z_bed::M
    z_surf::M
    rotation::M
    domain::AABB{T}
    rotated_domain::AABB{T}
end

struct SyntheticElevation{T,B,S} <: AbstractElevation{T}
    z_bed::B
    z_surf::S
    domain::AABB{T}
end

domain(dem::SyntheticElevation) = dem.domain
rotated_domain(dem::AbstractElevation) = domain(dem)
rotation(::AbstractElevation) = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

domain(dem::DataElevation) = dem.domain
rotated_domain(dem::DataElevation) = dem.rotated_domain
rotation(dem::DataElevation) = dem.rotation

"Construct AABB from coordinates"
AABB(xs, ys, zs) = AABB(extrema(xs)..., extrema(ys)..., extrema(zs)...)

"AABB extents"
extents(box::AABB) = box.xmax - box.xmin, box.ymax - box.ymin, box.zmax - box.zmin

"AABB center"
function center(box::AABB{T}) where {T}
    half = convert(T, 0.5)
    return half * (box.xmin + box.xmax), half * (box.ymin + box.ymax), half * (box.zmin + box.zmax)
end

"Dilate AABB by extending its limits around the center by certain fraction in each dimension"
function dilate(box::AABB, fractions)
    Δx, Δy, Δz = extents(box) .* fractions
    return AABB(box.xmin - Δx, box.xmax + Δx, box.ymin - Δy, box.ymax + Δy, box.zmin - Δz, box.zmax + Δz)
end

"Filter NaNs."
filtered(X) = filter(_x -> !isnan(_x), X)

"Create AABB enclosing both box1 and box2"
function union(box1::AABB, box2::AABB)
    return AABB(min(box1.xmin, box2.xmin), max(box1.xmax, box2.xmax),
                min(box1.ymin, box2.ymin), max(box1.ymax, box2.ymax),
                min(box1.zmin, box2.zmin), max(box1.zmax, box2.zmax))
end

"Rotate field `X`, `Y`, `Z` with rotation matrix `R`."
function rotate(X, Y, Z, R)
    xrot = R[1, 1] .* X .+ R[1, 2] .* Y .+ R[1, 3] .* Z
    yrot = R[2, 1] .* X .+ R[2, 2] .* Y .+ R[2, 3] .* Z
    zrot = R[3, 1] .* X .+ R[3, 2] .* Y .+ R[3, 3] .* Z
    return xrot, yrot, zrot
end

"Rotate field `X`, `Y`, `Z` with rotation matrix `R` and return extents."
rotate_minmax(X, Y, Z, R) = rotate(collect(extrema(X)), collect(extrema(Y)), collect(extrema(filtered(Z))), R)

"Generate SyntheticElevation data."
function generate_elevation(lx, ly, zmin, zmax, z_bed, z_surf)
    domain = AABB(-lx / 2, lx / 2, -ly / 2, ly / 2, zmin, zmax)
    return SyntheticElevation(z_bed, z_surf, domain)
end

"Generate DataElevation data."
function DataElevation(x, y, offsets, z_bed, z_surf, R)
    # get non-rotated domain
    domain = AABB(extrema(x)..., extrema(y)..., Float64(minimum([minimum(filtered(z_bed)), minimum(filtered(z_surf))])),
                  Float64(maximum([maximum(filtered(z_bed)), maximum(filtered(z_surf))])))
    # rotate bed and surface
    bed_extents = AABB(rotate_minmax(x, y, z_bed, R)...)
    surf_extents = AABB(rotate_minmax(x, y, z_surf, R)...)
    # get rotated domain
    rotated_domain = union(bed_extents, surf_extents)
    dem = DataElevation(x, y, offsets, z_bed, z_surf, R, domain, rotated_domain)
    return dem
end

end # module
