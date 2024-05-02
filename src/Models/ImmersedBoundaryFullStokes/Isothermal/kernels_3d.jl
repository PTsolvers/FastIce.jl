# pseudo-transient update rules

@kernel inbounds = true function update_σ!(Pr, τ, V, η, ω, Δτ, g::StructuredGrid{3}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    if isvalid(ω.na, location(Pr), I) && !isempty(ω.ns, location(Pr), I)
        # strain rates
        ε̇xx = ∂x(V.x, ω.ns, g, I)
        ε̇yy = ∂y(V.y, ω.ns, g, I)
        ε̇zz = ∂z(V.z, ω.ns, g, I)
        # velocity divergence
        ∇V = ε̇xx + ε̇yy + ε̇zz
        # hydrostatic stress
        Pr[I] -= ∇V * lerp(η, location(Pr), g, I) * Δτ.Pr
        # deviatoric diagonal
        τ.xx[I] -= (τ.xx[I] - 2.0 * lerp(η, location(τ.xx), g, I) * (ε̇xx - ∇V / 3.0)) * Δτ.τ.xx
        τ.yy[I] -= (τ.yy[I] - 2.0 * lerp(η, location(τ.yy), g, I) * (ε̇yy - ∇V / 3.0)) * Δτ.τ.yy
        τ.zz[I] -= (τ.zz[I] - 2.0 * lerp(η, location(τ.zz), g, I) * (ε̇zz - ∇V / 3.0)) * Δτ.τ.zz
    else
        Pr[I]   = 0.0
        τ.xx[I] = 0.0
        τ.yy[I] = 0.0
        τ.zz[I] = 0.0
    end
    # deviatoric off-diagonal
    if isvalid(ω.na, location(τ.xy), I) && !isempty(ω.ns, location(τ.xy), I)
        ε̇xy = 0.5 * (∂x(V.y, ω.ns, g, I) + ∂y(V.x, ω.ns, g, I))
        τ.xy[I] -= (τ.xy[I] - 2.0 * lerp(η, location(τ.xy), g, I) * ε̇xy) * Δτ.τ.xy
    else
        τ.xy[I] = 0.0
    end
    if isvalid(ω.na, location(τ.xz), I) && !isempty(ω.ns, location(τ.xz), I)
        ε̇xz = 0.5 * (∂x(V.z, ω.ns, g, I) + ∂z(V.x, ω.ns, g, I))
        τ.xz[I] -= (τ.xz[I] - 2.0 * lerp(η, location(τ.xz), g, I) * ε̇xz) * Δτ.τ.xz
    else
        τ.xz[I] = 0.0
    end
    if isvalid(ω.na, location(τ.yz), I) && !isempty(ω.ns, location(τ.yz), I)
        ε̇yz = 0.5 * (∂y(V.z, ω.ns, g, I) + ∂z(V.y, ω.ns, g, I))
        τ.yz[I] -= (τ.yz[I] - 2.0 * lerp(η, location(τ.yz), g, I) * ε̇yz) * Δτ.τ.yz
    else
        τ.yz[I] = 0.0
    end
end

@kernel inbounds = true function update_V!(V, Pr, τ, η, η_next, rheology, ρg, ω, Δτ, g::StructuredGrid{3}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # velocity
    if isvalid(ω.ns, location(V.x), I) && !isempty(ω.na, location(V.x), I)
        V.x[I] += (-∂x(Pr, ω.na, g, I) + ∂x(τ.xx, ω.na, g, I) + ∂y(τ.xy, ω.na, g, I) + ∂z(τ.xz, ω.na, g, I) - ρg.x[I]) / lerp(η, location(V.x), g, I) /
                  at(ω.ns, location(V.x), I) * Δτ.V.x
    else
        V.x[I] = 0.0
    end
    if isvalid(ω.ns, location(V.y), I) && !isempty(ω.na, location(V.y), I)
        V.y[I] += (-∂y(Pr, ω.na, g, I) + ∂y(τ.yy, ω.na, g, I) + ∂x(τ.xy, ω.na, g, I) + ∂z(τ.yz, ω.na, g, I) - ρg.y[I]) / lerp(η, location(V.y), g, I) /
                  at(ω.ns, location(V.y), I) * Δτ.V.y
    else
        V.y[I] = 0.0
    end
    if isvalid(ω.ns, location(V.z), I) && !isempty(ω.na, location(V.z), I)
        V.z[I] += (-∂z(Pr, ω.na, g, I) + ∂z(τ.zz, ω.na, g, I) + ∂x(τ.xz, ω.na, g, I) + ∂y(τ.yz, ω.na, g, I) - ρg.z[I]) / lerp(η, location(V.z), g, I) /
                  at(ω.ns, location(V.z), I) * Δτ.V.z
    else
        V.z[I] = 0.0
    end
    # rheology
    τII = sqrt(0.5 * (τ.xx[I]^2 + τ.yy[I]^2 + τ.zz[I]^2) +
               lerp(τ.xy, location(η), g, I)^2 +
               lerp(τ.xz, location(η), g, I)^2 +
               lerp(τ.yz, location(η), g, I)^2)
    η_next[I] = rheology(τII, I)
end

# helper kernels

@kernel inbounds = true function compute_τ!(τ, V, η, ω, g::StructuredGrid{3}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    if isvalid(ω.na, location(Pr), I) && !isempty(ω.ns, location(Pr), I)
        # strain rates
        ε̇xx = ∂x(V.x, ω.ns, g, I)
        ε̇yy = ∂y(V.y, ω.ns, g, I)
        ε̇zz = ∂z(V.z, ω.ns, g, I)
        # velocity divergence
        ∇V = ε̇xx + ε̇yy + ε̇zz
        # deviatoric diagonal
        τ.xx[I] = 2.0 * lerp(η, location(τ.xx), g, I) * (ε̇xx - ∇V / 3.0)
        τ.yy[I] = 2.0 * lerp(η, location(τ.yy), g, I) * (ε̇yy - ∇V / 3.0)
        τ.zz[I] = 2.0 * lerp(η, location(τ.zz), g, I) * (ε̇zz - ∇V / 3.0)
    else
        τ.xx[I] = 0.0
        τ.yy[I] = 0.0
        τ.zz[I] = 0.0
    end
    # deviatoric off-diagonal
    if isvalid(ω.na, location(τ.xy), I) && !isempty(ω.ns, location(τ.xy), I)
        ε̇xy = 0.5 * (∂x(V.y, ω.ns, g, I) + ∂y(V.x, ω.ns, g, I))
        τ.xy[I] = 2.0 * lerp(η, location(τ.xy), g, I) * ε̇xy
    else
        τ.xy[I] = 0.0
    end
    if isvalid(ω.na, location(τ.xz), I) && !isempty(ω.ns, location(τ.xz), I)
        ε̇xz = 0.5 * (∂x(V.z, ω.ns, g, I) + ∂z(V.x, ω.ns, g, I))
        τ.xz[I] = 2.0 * lerp(η, location(τ.xz), g, I) * ε̇xz
    else
        τ.xz[I] = 0.0
    end
    if isvalid(ω.na, location(τ.yz), I) && !isempty(ω.ns, location(τ.yz), I)
        ε̇yz = 0.5 * (∂y(V.z, ω.ns, g, I) + ∂z(V.y, ω.ns, g, I))
        τ.yz[I] = 2.0 * lerp(η, location(τ.yz), g, I) * ε̇yz
    else
        τ.yz[I] = 0.0
    end
end

@kernel inbounds = true function compute_residuals!(r_V, r_Pr, Pr, τ, V, ρg, ω, g::StructuredGrid{3}, O=Offset())
    I = @index(Global, Cartesian)
    I += O
    # pressure
    if isvalid(ω.na, location(Pr), I) && !isempty(ω.ns, location(Pr), I)
        r_Pr[I] = ∂x(V.x, ω.ns, g, I) + ∂y(V.y, ω.ns, g, I) + ∂z(V.z, ω.ns, g, I)
    else
        r_Pr[I] = 0.0
    end
    # velocity
    if isvalid(ω.ns, location(V.x), I) && !isempty(ω.na, location(V.x), I)
        r_V.x[I] = -∂x(Pr, ω.na, g, I) + ∂x(τ.xx, ω.na, g, I) + ∂y(τ.xy, ω.na, g, I) + ∂z(τ.xz, ω.na, g, I) - ρg.x[I]
    else
        r_V.x[I] = 0.0
    end
    if isvalid(ω.ns, location(V.y), I) && !isempty(ω.na, location(V.y), I)
        r_V.y[I] = -∂y(Pr, ω.na, g, I) + ∂y(τ.yy, ω.na, g, I) + ∂x(τ.xy, ω.na, g, I) + ∂z(τ.yz, ω.na, g, I) - ρg.y[I]
    else
        r_V.y[I] = 0.0
    end
    if isvalid(ω.ns, location(V.z), I) && !isempty(ω.na, location(V.z), I)
        r_V.z[I] = -∂z(Pr, ω.na, g, I) + ∂z(τ.zz, ω.na, g, I) + ∂x(τ.xz, ω.na, g, I) + ∂y(τ.yz, ω.na, g, I) - ρg.z[I]
    else
        r_V.z[I] = 0.0
    end
end
