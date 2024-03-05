# pseudo-transient update rules

@kernel inbounds = true function update_σ!(Pr, τ, V, η, Δτ, g::CartesianGrid{3}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # strain rates
    ε̇xx = ∂x(V.x, g, I)
    ε̇yy = ∂y(V.y, g, I)
    ε̇zz = ∂z(V.z, g, I)
    ε̇xy = 0.5 * (∂x(V.y, g, I) + ∂y(V.x, g, I))
    ε̇xz = 0.5 * (∂x(V.z, g, I) + ∂z(V.x, g, I))
    ε̇yz = 0.5 * (∂y(V.z, g, I) + ∂z(V.y, g, I))
    # velocity divergence
    ∇V = ε̇xx + ε̇yy + ε̇zz
    # hydrostatic stress
    Pr[I] -= ∇V * η[I] * Δτ.Pr
    # deviatoric diagonal
    τ.xx[I] -= (τ.xx[I] - 2.0 * itp(η, location(τ.xx), g, I) * (ε̇xx - ∇V / 3.0)) * Δτ.τ.xx
    τ.yy[I] -= (τ.yy[I] - 2.0 * itp(η, location(τ.yy), g, I) * (ε̇yy - ∇V / 3.0)) * Δτ.τ.yy
    τ.zz[I] -= (τ.zz[I] - 2.0 * itp(η, location(τ.zz), g, I) * (ε̇zz - ∇V / 3.0)) * Δτ.τ.zz
    # deviatoric off-diagonal
    τ.xy[I] -= (τ.xy[I] - 2.0 * itp(η, location(τ.xy), g, I) * ε̇xy) * Δτ.τ.xy
    τ.xz[I] -= (τ.xz[I] - 2.0 * itp(η, location(τ.xz), g, I) * ε̇xz) * Δτ.τ.xz
    τ.yz[I] -= (τ.yz[I] - 2.0 * itp(η, location(τ.yz), g, I) * ε̇yz) * Δτ.τ.yz
end

@kernel inbounds = true function update_V!(V, Pr, τ, η, η_next, rheology, ρg, Δτ, g::CartesianGrid{3}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # velocity
    V.x[I] += (-∂x(Pr, g, I) + ∂x(τ.xx, g, I) + ∂y(τ.xy, g, I) + ∂z(τ.xz, g, I) - ρg.x[I]) / itp(η, location(V.x), g, I) * Δτ.V.x
    V.y[I] += (-∂y(Pr, g, I) + ∂y(τ.yy, g, I) + ∂x(τ.xy, g, I) + ∂z(τ.yz, g, I) - ρg.y[I]) / itp(η, location(V.y), g, I) * Δτ.V.y
    V.z[I] += (-∂z(Pr, g, I) + ∂z(τ.zz, g, I) + ∂x(τ.xz, g, I) + ∂y(τ.yz, g, I) - ρg.z[I]) / itp(η, location(V.z), g, I) * Δτ.V.z
    # rheology
    τII = sqrt(0.5 * (τ.xx[I]^2 + τ.yy[I]^2 + τ.zz[I]^2) +
               itp(τ.xy, location(η), g, I)^2 +
               itp(τ.xz, location(η), g, I)^2 +
               itp(τ.yz, location(η), g, I)^2)
    η_next[I] = rheology(τII, I)
end

# helper kernels

@kernel inbounds = true function compute_τ!(τ, V, η, g::CartesianGrid{3}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # strain rates
    ε̇xx = ∂x(V.x, g, I)
    ε̇yy = ∂y(V.y, g, I)
    ε̇zz = ∂z(V.z, g, I)
    ε̇xy = 0.5 * (∂x(V.y, g, I) + ∂y(V.x, g, I))
    ε̇xz = 0.5 * (∂x(V.z, g, I) + ∂z(V.x, g, I))
    ε̇yz = 0.5 * (∂y(V.z, g, I) + ∂z(V.y, g, I))
    # velocity divergence
    ∇V = ε̇xx + ε̇yy + ε̇zz
    # deviatoric diagonal
    τ.xx[I] = 2.0 * itp(η, location(τ.xx), g, I) * (ε̇xx - ∇V / 3.0)
    τ.yy[I] = 2.0 * itp(η, location(τ.yy), g, I) * (ε̇yy - ∇V / 3.0)
    τ.zz[I] = 2.0 * itp(η, location(τ.zz), g, I) * (ε̇zz - ∇V / 3.0)
    # deviatoric off-diagonal
    τ.xy[I] = 2.0 * itp(η, location(τ.xy), g, I) * ε̇xy
    τ.xz[I] = 2.0 * itp(η, location(τ.xz), g, I) * ε̇xz
    τ.yz[I] = 2.0 * itp(η, location(τ.yz), g, I) * ε̇yz
end

@kernel inbounds = true function compute_residuals!(r_V, r_Pr, Pr, τ, V, ρg, g::CartesianGrid{3}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # pressure
    r_Pr[I] = ∂x(V.x, g, I) + ∂y(V.y, g, I) + ∂z(V.z, g, I)
    # velocity
    r_V.x[I] = -∂x(Pr, g, I) + ∂x(τ.xx, g, I) + ∂y(τ.xy, g, I) + ∂z(τ.xz, g, I) - ρg.x[I]
    r_V.y[I] = -∂y(Pr, g, I) + ∂y(τ.yy, g, I) + ∂x(τ.xy, g, I) + ∂z(τ.yz, g, I) - ρg.y[I]
    r_V.z[I] = -∂z(Pr, g, I) + ∂z(τ.zz, g, I) + ∂x(τ.xz, g, I) + ∂y(τ.yz, g, I) - ρg.z[I]
end
