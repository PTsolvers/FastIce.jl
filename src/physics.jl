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

default(::Type{IncompressibleIceEOS{T}}) where {T} = IncompressibleIceEOS(convert(T, 920), convert(T, 2100))

struct IceThermalProperties{T}
    thermal_conductivity::T
    melting_temperature::T
end

default(::Type{IceThermalProperties{T}}) where {T} = IceThermalProperties(convert(T, 1), convert(T, 273.15))

abstract type IceRheology end

struct GlensLawRheology{T,I}
    consistency::T
    exponent::I
end

default(::Type{GlensLawRheology{T,I}}) where {T,I} = GlensLawRheology(convert(T, 2.4e-24), convert(I, 3))

@inline function (rh::GlensLawRheology{T})(τII::T) where {T}
    return 0.5 / (rh.consistency * τII^(rh.exponent - 1))
end

end