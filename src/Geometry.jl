module Geometry

export AbstractElevation, DataElevation, AABB

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

struct AABB{T<:Union{Real}}
    xmin::T
    xmax::T
    ymin::T
    ymax::T
    zmin::T
    zmax::T
end

end
