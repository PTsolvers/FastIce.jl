using KernelAbstractions

using FastIce.GridOperators

@kernel function update_η!(η, η_rh, χ, grid, fields, args...)
    I = @index(Global, Cartesian)
    ηt = η_rh(grid, I, fields, args...)
    @inbounds η[I] = exp(log(η[I]) * (1 - χ) + log(ηt) * χ)
end

@kernel function update_σ!(Pr, τ, V, η, Δτ, Δ)
    I = @index(Global, Cartesian)
    @inbounds if checkbounds(Bool, Pr, I)
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
    @inbounds if checkbounds(Bool, τ.xy, I)
        ε̇xy = 0.5 * (∂ᵛx(V.y, I) / Δ.x + ∂ᵛy(V.x, I) / Δ.y)
        τ.xy[I] -= (τ.xy[I] - 2.0 * avᵛxy(η, I) * ε̇xy) * Δτ.τ.xy
    end
    @inbounds if checkbounds(Bool, τ.xz, I)
        ε̇xz = 0.5 * (∂ᵛx(V.z, I) / Δ.x + ∂ᵛz(V.x, I) / Δ.z)
        τ.xz[I] -= (τ.xz[I] - 2.0 * avᵛxz(η, I) * ε̇xz) * Δτ.τ.xz
    end
    @inbounds if checkbounds(Bool, τ.yz, I)
        ε̇yz = 0.5 * (∂ᵛy(V.z, I) / Δ.y + ∂ᵛz(V.y, I) / Δ.z)
        τ.yz[I] -= (τ.yz[I] - 2.0 * avᵛyz(η, I) * ε̇yz) * Δτ.τ.yz
    end
end

@kernel function update_V!(V, Pr, τ, η, Δτ, Δ)
    I = @index(Global, Cartesian)
    @inbounds if checkbounds(Bool, V.x, I)
        ∂σxx_∂x = (-∂ᵛx(Pr, I) + ∂ᵛx(τ.xx, I)) / Δ.x
        ∂τxy_∂y = ∂ᶜy(τ.xy, I) / Δ.y
        ∂τxz_∂z = ∂ᶜz(τ.xz, I) / Δ.z
        V.x[I] += (∂σxx_∂x + ∂τxy_∂y + ∂τxz_∂z) / maxlᵛx(η, I) * Δτ.V.x
    end
    @inbounds if checkbounds(Bool, V.y, I)
        ∂σyy_∂y = (-∂ᵛy(Pr, I) + ∂ᵛy(τ.yy, I)) / Δ.y
        ∂τxy_∂x = ∂ᶜx(τ.xy, I) / Δ.x
        ∂τyz_∂z = ∂ᶜz(τ.yz, I) / Δ.z
        V.y[I] += (∂σyy_∂y + ∂τxy_∂x + ∂τyz_∂z) / maxlᵛy(η, I) * Δτ.V.y
    end
    @inbounds if checkbounds(Bool, V.z, I)
        ∂σzz_∂z = (-∂ᵛz(Pr, I) + ∂ᵛz(τ.zz, I)) / Δ.z
        ∂τxz_∂x = ∂ᶜx(τ.xz, I) / Δ.x
        ∂τyz_∂y = ∂ᶜy(τ.yz, I) / Δ.y
        V.z[I] += (∂σzz_∂z + ∂τxz_∂x + ∂τyz_∂y) / maxlᵛz(η, I) * Δτ.V.z
    end
end