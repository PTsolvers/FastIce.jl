# pseudo-transient update rules

@kernel inbounds = true function update_σ!(Pr, τ, V, η, Δτ, Δ, grid::CartesianGrid{3}, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    if checkbounds(Bool, Pr, I)
        ε̇xx = ∂ᶜx(V.x, I) / Δ.x
        ε̇yy = ∂ᶜy(V.y, I) / Δ.y
        ε̇zz = ∂ᶜz(V.z, I) / Δ.z
        ∇V = ε̇xx + ε̇yy + ε̇zz
        # hydrostatic
        Pr[I] -= ∇V * η[I] * Δτ.Pr
        # deviatoric diagonal
        τ.xx[I] -= (τ.xx[I] - 2.0 * η[I] * (ε̇xx - ∇V / 3.0)) * Δτ.τ.xx
        τ.yy[I] -= (τ.yy[I] - 2.0 * η[I] * (ε̇yy - ∇V / 3.0)) * Δτ.τ.yy
        τ.zz[I] -= (τ.zz[I] - 2.0 * η[I] * (ε̇zz - ∇V / 3.0)) * Δτ.τ.zz
    end
    # deviatoric off-diagonal
    if checkbounds(Bool, τ.xy, I)
        τ.xy[I] -= (τ.xy[I] - avᵛxy(η, I) * (∂ᵛx(V.y, I) / Δ.x + ∂ᵛy(V.x, I) / Δ.y)) * Δτ.τ.xy
    end
    if checkbounds(Bool, τ.xz, I)
        τ.xz[I] -= (τ.xz[I] - avᵛxz(η, I) * (∂ᵛx(V.z, I) / Δ.x + ∂ᵛz(V.x, I) / Δ.z)) * Δτ.τ.xz
    end
    if checkbounds(Bool, τ.yz, I)
        τ.yz[I] -= (τ.yz[I] - avᵛyz(η, I) * (∂ᵛy(V.z, I) / Δ.y + ∂ᵛz(V.y, I) / Δ.z)) * Δτ.τ.yz
    end
end

@kernel inbounds = true function update_V!(V, Pr, τ, η, η_next, viscosity, ρg, Δτ, Δ, grid::CartesianGrid{3}, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    if within(grid, V.x, I)
        ∂σxx_∂x = (-∂ᵛx(Pr, I) + ∂ᵛx(τ.xx, I)) / Δ.x
        ∂τxy_∂y = ∂ᶜy(τ.xy, I) / Δ.y
        ∂τxz_∂z = ∂ᶜz(τ.xz, I) / Δ.z
        V.x[I] += (∂σxx_∂x + ∂τxy_∂y + ∂τxz_∂z - ρg.x[I]) / maxlᵛx(η, I) * Δτ.V.x
    end
    if within(grid, V.y, I)
        ∂σyy_∂y = (-∂ᵛy(Pr, I) + ∂ᵛy(τ.yy, I)) / Δ.y
        ∂τxy_∂x = ∂ᶜx(τ.xy, I) / Δ.x
        ∂τyz_∂z = ∂ᶜz(τ.yz, I) / Δ.z
        V.y[I] += (∂σyy_∂y + ∂τxy_∂x + ∂τyz_∂z - ρg.y[I]) / maxlᵛy(η, I) * Δτ.V.y
    end
    if within(grid, V.z, I)
        ∂σzz_∂z = (-∂ᵛz(Pr, I) + ∂ᵛz(τ.zz, I)) / Δ.z
        ∂τxz_∂x = ∂ᶜx(τ.xz, I) / Δ.x
        ∂τyz_∂y = ∂ᶜy(τ.yz, I) / Δ.y
        V.z[I] += (∂σzz_∂z + ∂τxz_∂x + ∂τyz_∂y - ρg.z[I]) / maxlᵛz(η, I) * Δτ.V.z
    end
    # update viscosity
    if within(grid, η_next, I)
        τII = sqrt(0.5 * (τ.xx[I]^2 + τ.yy[I]^2 + τ.zz[I]^2) +
                   avᶜxy(τ.xy, I)^2 + avᶜxz(τ.xz, I)^2 + avᶜyz(τ.yz, I)^2)
        η_next[I] = viscosity(τII, I)
    end
end

# helper kernels

@kernel inbounds = true function compute_τ!(τ, V, η, Δ, grid::CartesianGrid{3}, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    if checkbounds(Bool, Pr, I)
        ε̇xx = ∂ᶜx(V.x, I) / Δ.x
        ε̇yy = ∂ᶜy(V.y, I) / Δ.y
        ε̇zz = ∂ᶜz(V.z, I) / Δ.z
        ∇V = ε̇xx + ε̇yy + ε̇zz
        # deviatoric diagonal
        τ.xx[I] = 2.0 * η[I] * (ε̇xx - ∇V / 3.0)
        τ.yy[I] = 2.0 * η[I] * (ε̇yy - ∇V / 3.0)
        τ.zz[I] = 2.0 * η[I] * (ε̇zz - ∇V / 3.0)
    end
    if checkbounds(Bool, τ.xy, I)
        τ.xy[I] = avᵛxy(η, I) * (∂ᵛx(V.y, I) / Δ.x + ∂ᵛy(V.x, I) / Δ.y)
    end
    if checkbounds(Bool, τ.xz, I)
        τ.xz[I] = avᵛxz(η, I) * (∂ᵛx(V.z, I) / Δ.x + ∂ᵛz(V.x, I) / Δ.z)
    end
    if checkbounds(Bool, τ.yz, I)
        τ.yz[I] = avᵛyz(η, I) * (∂ᵛy(V.z, I) / Δ.y + ∂ᵛz(V.y, I) / Δ.z)
    end
end

@kernel inbounds = true function compute_residuals!(r_V, r_Pr, Pr, τ, V, ρg, Δ, grid::CartesianGrid{3}, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    if within(grid, r_Pr, I)
        r_Pr[I] = ∂ᶜx(V.x, I) / Δ.x + ∂ᶜy(V.y, I) / Δ.y + ∂ᶜz(V.z, I) / Δ.z
    end
    if within(grid, r_V.x, I)
        ∂σxx_∂x = (-∂ᵛx(Pr, I) + ∂ᵛx(τ.xx, I)) / Δ.x
        ∂τxy_∂y = ∂ᶜy(τ.xy, I) / Δ.y
        ∂τxz_∂z = ∂ᶜz(τ.xz, I) / Δ.z
        r_V.x[I] = ∂σxx_∂x + ∂τxy_∂y + ∂τxz_∂z - ρg.x[I]
    end
    if within(grid, r_V.y, I)
        ∂σyy_∂y = (-∂ᵛy(Pr, I) + ∂ᵛy(τ.yy, I)) / Δ.y
        ∂τxy_∂x = ∂ᶜx(τ.xy, I) / Δ.x
        ∂τyz_∂z = ∂ᶜz(τ.yz, I) / Δ.z
        r_V.y[I] = ∂σyy_∂y + ∂τxy_∂x + ∂τyz_∂z - ρg.y[I]
    end
    if within(grid, r_V.z, I)
        ∂σzz_∂z = (-∂ᵛz(Pr, I) + ∂ᵛz(τ.zz, I)) / Δ.z
        ∂τxz_∂x = ∂ᶜx(τ.xz, I) / Δ.x
        ∂τyz_∂y = ∂ᶜy(τ.yz, I) / Δ.y
        r_V.z[I] = ∂σzz_∂z + ∂τxz_∂x + ∂τyz_∂y - ρg.z[I]
    end
end
