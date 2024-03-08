module Geometry

export AbstractElevation, AABB, DataElevation, SyntheticElevation
export domain, rotated_domain, rotation, generate_elevation, extents, center, dilate

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

"Construct AABB from coordinates"
AABB(xs, ys, zs) = AABB(extrema(xs)..., extrema(ys)..., extrema(zs)...)

"Generate SyntheticElevation data."
function generate_elevation(lx, ly, zmin, zmax, z_bed, z_surf)
    domain = AABB(-lx / 2, lx / 2, -ly / 2, ly / 2, zmin, zmax)
    return SyntheticElevation(z_bed, z_surf, domain)
end

"AABB extents"
function extents(box::AABB)
    return box.xmax - box.xmin, box.ymax - box.ymin, box.zmax - box.zmin
end

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

end # module
