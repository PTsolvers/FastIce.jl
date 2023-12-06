module Physics

export IceRheology, PowerLawRheology, Relaxation, ViscoplasticRegularisation

abstract type IceRheology end

struct LinearViscousRheology{V} <: IceRheology
    η::V
end

@inline (l::LinearViscousRheology)(τII, I) = l.η[I]

struct PowerLawRheology{F,P<:Real} <: IceRheology
    A::F
    n::P
end

Base.@propagate_inbounds (r::PowerLawRheology)(τII, I) = 0.5 / (r.A[I] * τII^(r.n - 1))

struct ViscoplasticRegularisation{V,E} <: IceRheology
    viscosity::V
    inv_η_reg::E
    ViscoplasticRegularisation(viscosity::V, η_reg::E) where {V,E} = new{V,E}(viscosity, one(η_reg) / η_reg)
end

Base.@propagate_inbounds (r::ViscoplasticRegularisation)(τII, I) = 1.0 / (1.0 / r.viscosity(τII, I) + r.inv_η_reg)

struct Relaxation{R,T,F,S,IS} <: IceRheology
    field::F
    target::T
    rate::R
    scale::S
    inv_scale::IS
end

function Relaxation(field, target, rate; scale=nothing)
    inv_scale = isnothing(scale) ? nothing :
                scale == log     ? exp     :
                scale == log10   ? exp10   :
                throw(ArgumentError("only `nothing`, `log` and `log10` are supported for the relaxation scale"))
    return Relaxation(field, target, rate, scale, inv_scale)
end

const LinSpaceRelaxation{R,T,F} = Relaxation{R,T,F,Nothing,Nothing}

Base.@propagate_inbounds (r::LinSpaceRelaxation)(τII, I) = r.field[I] * (1 - r.rate) + r.target(τII, I) * r.rate
Base.@propagate_inbounds (r::Relaxation)(τII, I) = r.inv_scale(r.scale(r.field[I]) * (1 - r.rate) + r.scale(r.target(τII, I)) * r.rate)

end
