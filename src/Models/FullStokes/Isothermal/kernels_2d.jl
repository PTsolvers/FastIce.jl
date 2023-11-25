# pseudo-transient update rules

@kernel function update_σ!(Pr, τ, V, η, Δτ, Δ, grid::CartesianGrid{2}, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    @inbounds if checkbounds(Bool, Pr, I)
        ε̇xx = ∂ᶜx(V.x, I) / Δ.x
        ε̇yy = ∂ᶜy(V.y, I) / Δ.y
        ∇V = ε̇xx + ε̇yy
        # hydrostatic
        Pr[I] -= ∇V * η[I] * Δτ.Pr
        # deviatoric diagonal
        τ.xx[I] -= (τ.xx[I] - 2.0 * η[I] * (ε̇xx - ∇V / 3.0)) * Δτ.τ.xx
        τ.yy[I] -= (τ.yy[I] - 2.0 * η[I] * (ε̇yy - ∇V / 3.0)) * Δτ.τ.yy
    end
    @inbounds if checkbounds(Bool, τ.xy, I)
        ε̇xy = 0.5 * (∂ᵛx(V.y, I) / Δ.x + ∂ᵛy(V.x, I) / Δ.y)
        τ.xy[I] -= (τ.xy[I] - 2.0 * avᵛxy(η, I) * ε̇xy) * Δτ.τ.xy
    end
end

@kernel function update_V!(V, Pr, τ, η, ρg, Δτ, Δ, grid::CartesianGrid{2}, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    @inbounds if within(grid, V.x, I)
        ∂σxx_∂x = (-∂ᵛx(Pr, I) + ∂ᵛx(τ.xx, I)) / Δ.x
        ∂τxy_∂y = ∂ᶜy(τ.xy, I) / Δ.y
        V.x[I] += (∂σxx_∂x + ∂τxy_∂y - ρg.x[I]) / maxlᵛx(η, I) * Δτ.V.x
    end
    @inbounds if within(grid, V.y, I)
        ∂σyy_∂y = (-∂ᵛy(Pr, I) + ∂ᵛy(τ.yy, I)) / Δ.y
        ∂τxy_∂x = ∂ᶜx(τ.xy, I) / Δ.x
        V.y[I] += (∂σyy_∂y + ∂τxy_∂x - ρg.y[I]) / maxlᵛy(η, I) * Δτ.V.y
    end
end

# helper kernels

@kernel function compute_τ!(τ, V, η, Δ, grid::CartesianGrid{2}, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    @inbounds if checkbounds(Bool, Pr, I)
        ε̇xx = ∂ᶜx(V.x, I) / Δ.x
        ε̇yy = ∂ᶜy(V.y, I) / Δ.y
        ∇V = ε̇xx + ε̇yy
        # deviatoric diagonal
        τ.xx[I] =  2.0 * η[I] * (ε̇xx - ∇V / 3.0)
        τ.yy[I] =  2.0 * η[I] * (ε̇yy - ∇V / 3.0)
    end
    @inbounds if checkbounds(Bool, τ.xy, I)
        ε̇xy = 0.5 * (∂ᵛx(V.y, I) / Δ.x + ∂ᵛy(V.x, I) / Δ.y)
        τ.xy[I] = 2.0 * avᵛxy(η, I) * ε̇xy
    end
end

@kernel function compute_residuals!(r_V, r_Pr, Pr, τ, V, ρg, Δ, grid::CartesianGrid{2}, offset=nothing)
    I = @index(Global, Cartesian)
    isnothing(offset) || (I += offset)
    @inbounds if within(grid, r_Pr, I)
        ε̇xx = ∂ᶜx(V.x, I) / Δ.x
        ε̇yy = ∂ᶜy(V.y, I) / Δ.y
        r_Pr[I] = ε̇xx + ε̇yy
    end
    @inbounds if within(grid, r_V.x, I)
        ∂σxx_∂x = (-∂ᵛx(Pr, I) + ∂ᵛx(τ.xx, I)) / Δ.x
        ∂τxy_∂y = ∂ᶜy(τ.xy, I) / Δ.y
        r_V.x[I] = ∂σxx_∂x + ∂τxy_∂y - ρg.x[I]
    end
    @inbounds if within(grid, r_V.y, I)
        ∂σyy_∂y = (-∂ᵛy(Pr, I) + ∂ᵛy(τ.yy, I)) / Δ.y
        ∂τxy_∂x = ∂ᶜx(τ.xy, I) / Δ.x
        r_V.y[I] = ∂σyy_∂y + ∂τxy_∂x - ρg.y[I]
    end
end
