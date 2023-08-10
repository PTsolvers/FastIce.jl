module Physics

export AbstractPhysics
export IncompressibleIceEOS, IceThermalProperties
export IceRheology, GlensLawRheology, τ
export default

using FastIce.Macros

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

@inline function (rh::GlensLawRheology{T})(grid, ix, iy, iz, fields) where {T}
    (; τ, A) = fields
    @inbounds τII = sqrt(0.5 * (@inn(τ.xx)^2 + @inn(τ.yy)^2 + @inn(τ.zz)^2) + @av_xy(τ.xy)^2 + @av_xz(τ.xz)^2 + @av_yz(τ.yz)^2)
    return 0.5 / (A[ix, iy, iz] * τII^(rh.exponent - 1))
end

end