# pseudo-transient update rules

@kernel inbounds = true function update_σ!(Pr, τ, V, η, Δτ, g::StructuredGrid{2}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # strain rates
    ε̇xx = ∂x(V.x, g, I)
    ε̇yy = ∂y(V.y, g, I)
    ε̇xy = 0.5 * (∂x(V.y, g, I) + ∂y(V.x, g, I))
    # velocity divergence
    ∇V = ε̇xx + ε̇yy
    # hydrostatic stress
    Pr[I] -= ∇V * η[I] * Δτ.Pr
    # deviatoric stress
    τ.xx[I] -= (τ.xx[I] - 2.0 * itp(η, location(τ.xx), g, I) * (ε̇xx - ∇V / 3.0)) * Δτ.τ.xx
    τ.yy[I] -= (τ.yy[I] - 2.0 * itp(η, location(τ.yy), g, I) * (ε̇yy - ∇V / 3.0)) * Δτ.τ.yy
    τ.xy[I] -= (τ.xy[I] - 2.0 * itp(η, location(τ.xy), g, I) * ε̇xy) * Δτ.τ.xy
end

@kernel inbounds = true function update_V!(V, Pr, τ, η, η_next, rheology, ρg, Δτ, g::StructuredGrid{2}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # velocity
    V.x[I] += (-∂x(Pr, g, I) + ∂x(τ.xx, g, I) + ∂y(τ.xy, g, I) - ρg.x[I]) / itp(η, location(V.x), I) * Δτ.V.x
    V.y[I] += (-∂y(Pr, g, I) + ∂y(τ.yy, g, I) + ∂x(τ.xy, g, I) - ρg.y[I]) / itp(η, location(V.x), I) * Δτ.V.y
    # rheology
    τII       = sqrt(0.5 * (τ.xx[I]^2 + τ.yy[I]^2) + itp(τ.xy, location(η), g, I)^2)
    η_next[I] = rheology(τII, I)
end

# helper kernels

@kernel inbounds = true function compute_τ!(τ, V, η, g::CartesianGrid{2}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    ε̇xx = ∂x(V.x, g, I)
    ε̇yy = ∂y(V.y, g, I)
    ε̇xy = 0.5 * (∂x(V.y, g, I) + ∂y(V.x, g, I))
    ∇V = ε̇xx + ε̇yy
    # deviatoric diagonal
    τ.xx[I] = 2.0 * itp(η, location(τ.xx), g, I) * (ε̇xx - ∇V / 3.0)
    τ.yy[I] = 2.0 * itp(η, location(τ.yy), g, I) * (ε̇yy - ∇V / 3.0)
    τ.xy[I] = 2.0 * itp(η, location(τ.xy), g, I) * ε̇xy
end

@kernel inbounds = true function compute_residuals!(r_V, r_Pr, Pr, τ, V, ρg, g::CartesianGrid{2}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # pressure
    r_Pr[I] = ∂x(V.x, g, I) + ∂y(V.y, g, I)
    # velocity
    r_V.x[I] = -∂x(Pr, g, I) + ∂x(τ.xx, g, I) + ∂y(τ.xy, g, I) - ρg.x[I]
    r_V.y[I] = -∂y(Pr, g, I) + ∂y(τ.yy, g, I) + ∂x(τ.xy, g, I) - ρg.y[I]
end
