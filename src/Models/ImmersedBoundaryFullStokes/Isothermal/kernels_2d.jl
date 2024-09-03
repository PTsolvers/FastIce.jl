# pseudo-transient update rules

@kernel inbounds = true function update_σ!(Pr, τ, V, η, ω, Δτ, g::StructuredGrid{2}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # strain rates
    ε̇xx = ∂x(V.x, ω, g, I)
    ε̇yy = ∂y(V.y, ω, g, I)
    ε̇xy = 0.5 * (∂x(V.y, ω, g, I) + ∂y(V.x, ω, g, I))
    # velocity divergence
    ∇V = ε̇xx + ε̇yy
    # hydrostatic stress
    Pr[I] -= ∇V * lerp(η, location(Pr), g, I) * Δτ.Pr
    # deviatoric stress
    τ.xx[I] -= (τ.xx[I] - 2.0 * lerp(η, location(τ.xx), g, I) * (ε̇xx - ∇V / 3.0)) * Δτ.τ.xx
    τ.yy[I] -= (τ.yy[I] - 2.0 * lerp(η, location(τ.yy), g, I) * (ε̇yy - ∇V / 3.0)) * Δτ.τ.yy
    τ.xy[I] -= (τ.xy[I] - 2.0 * lerp(η, location(τ.xy), g, I) * ε̇xy) * Δτ.τ.xy
end

@kernel inbounds = true function update_V!(V, Pr, τ, η, η_next, rheology, ρg, ω, Δτ, g::StructuredGrid{2}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # velocity
    V.x[I] += (-∂x(Pr, ω, g, I) + ∂x(τ.xx, ω, g, I) + ∂y(τ.xy, ω, g, I) - at(ω, location(V.x), I) * ρg.x[I]) / lerp(η, location(V.x), I) * Δτ.V.x
    V.y[I] += (-∂y(Pr, ω, g, I) + ∂y(τ.yy, ω, g, I) + ∂x(τ.xy, ω, g, I) - at(ω, location(V.y), I) * ρg.y[I]) / lerp(η, location(V.x), I) * Δτ.V.y
    # rheology
    τII       = sqrt(0.5 * (τ.xx[I]^2 + τ.yy[I]^2) + lerp(τ.xy, location(η), g, I)^2)
    η_next[I] = rheology(τII, I)
end

# helper kernels

@kernel inbounds = true function compute_τ!(τ, V, η, ω, g::StructuredGrid{2}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    ε̇xx = ∂x(V.x, ω, g, I)
    ε̇yy = ∂y(V.y, ω, g, I)
    ε̇xy = 0.5 * (∂x(V.y, ω, g, I) + ∂y(V.x, ω, g, I))
    ∇V = ε̇xx + ε̇yy
    # deviatoric diagonal
    τ.xx[I] = 2.0 * lerp(η, location(τ.xx), g, I) * (ε̇xx - ∇V / 3.0)
    τ.yy[I] = 2.0 * lerp(η, location(τ.yy), g, I) * (ε̇yy - ∇V / 3.0)
    τ.xy[I] = 2.0 * lerp(η, location(τ.xy), g, I) * ε̇xy
end

@kernel inbounds = true function compute_residuals!(r_V, r_Pr, Pr, τ, V, ρg, ω, g::StructuredGrid{2}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # pressure
    r_Pr[I] = ∂x(V.x, ω, g, I) + ∂y(V.y, ω, g, I)
    # velocity
    r_V.x[I] = -∂x(Pr, ω, g, I) + ∂x(τ.xx, ω, g, I) + ∂y(τ.xy, ω, g, I) - at(ω, location(V.x), I) * ρg.x[I]
    r_V.y[I] = -∂y(Pr, ω, g, I) + ∂y(τ.yy, ω, g, I) + ∂x(τ.xy, ω, g, I) - at(ω, location(V.y), I) * ρg.y[I]
end
