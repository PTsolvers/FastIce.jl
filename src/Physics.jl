module Physics

export AbstractPhysics
export IncompressibleIceEOS, IceThermalProperties
export IceRheology, GlensLawRheology, τ
export default

using FastIce.GridOperators

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

struct GlensLawRheology{I}
    exponent::I
end

default(::Type{GlensLawRheology{I}}) where {I} = GlensLawRheology(convert(I, 3))

@inline function (rh::GlensLawRheology{T})(grid, I, fields) where {T}
    (; τ, A) = fields
    @inbounds τII = sqrt(0.5 * (τ.xx[I]^2 + τ.yy[I]^2 + τ.zz[I]^2) + avᶜxy(τ.xy, I)^2 + avᶜxz(τ.xz, I)^2 + avᶜyz(τ.yz, I)^2)
    return 0.5 / (A[I] * τII^(rh.exponent - 1))
end

end
