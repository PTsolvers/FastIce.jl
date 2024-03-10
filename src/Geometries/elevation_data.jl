abstract type AbstractElevation{T<:Real} end

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

struct SyntheticElevation{T,B,S} <: AbstractElevation{T}
    z_bed::B
    z_surf::S
    domain::AABB{T}
end

"Generate SyntheticElevation data."
function SyntheticElevation(lx, ly, zmin, zmax, z_bed, z_surf)
    domain = AABB(-lx / 2, lx / 2, -ly / 2, ly / 2, zmin, zmax)
    return SyntheticElevation(z_bed, z_surf, domain)
end

domain(dem::DataElevation) = dem.domain
rotated_domain(dem::DataElevation) = dem.rotated_domain
rotation(dem::DataElevation) = dem.rotation

domain(dem::SyntheticElevation) = dem.domain
rotated_domain(dem::AbstractElevation) = domain(dem)
rotation(::AbstractElevation) = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
