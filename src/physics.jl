module Physics

export AbstractPhysics
export IncompressibleIceEOS, IceThermalProperties
export IceRheology, GlensLawRheology, τ
export default

abstract type AbstractPhysics end

struct IncompressibleIceEOS{T}
    density::T
    heat_capacity::T
end

default(::Type{IncompressibleIceEOS{T}}) where T = IncompressibleIceEOS(convert(T, 920), convert(T, 2100))

struct IceThermalProperties{T}
    thermal_conductivity::T
    melting_temperature::T
end

abstract type IceRheology end

struct GlensLawRheology{T,I}
    consistency::T
    exponent::I
end

@inline function τ(rh::GlensLawRheology{T}, e::NamedTuple{(:xx, :yy, :xy),NTuple{2,T}}) where {T}
    γ̇ = sqrt(0.5 * (e.xx^2 + e.yy^2) + e.xy^2)
    η = rh.consistency * γ̇^(rh.exponent - 1)
    return (xx=η * e.xx, yy=η * e.yy, xy=η * e.xy)
end

@inline function τ(rh::GlensLawRheology{T}, e::NamedTuple{(:xx, :yy, :zz, :xy, :xz, :yz),NTuple{3,T}}) where {T}
    γ̇ = sqrt(0.5 * (e.xx^2 + e.yy^2 + +e.zz^2) + e.xy^2 + e.xz^2 + e.yz^2)
    η = rh.consistency * γ̇^(rh.exponent - 1)
    return (xx=η * e.xx, yy=η * e.yy, zz=η * e.zz, xy=η * e.xy, xz=η * e.xz, yz=η * e.yz)
end

end