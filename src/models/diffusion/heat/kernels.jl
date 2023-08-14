using KernelAbstractions

using FastIce.GridOperators

@kernel function update_σ!(T, q, Δτ, Δ)
    I = @index(Global, Cartesian)
    @inbounds if checkbounds(Bool, T, I)
        ∂qx_∂x = ∂ᶜx(q.x, I) / Δ.x
        ∂qy_∂y = ∂ᶜy(q.y, I) / Δ.y
        ∂qz_∂z = ∂ᶜz(q.z, I) / Δ.z
        ∇q = ∂qx_∂x + ∂qy_∂y + ∂qz_∂z
        T[I] -= ∇q * η[I] * Δτ.Pr
end

@kernel function update_q!(q, T, τ, η, Δτ, Δ)
    I = @index(Global, Cartesian)
    @inbounds if checkbounds(Bool, V.x, I)
        ∂T_∂x = -∂ᵛx(T, I) / Δ.x
        q.x[I] += ∂T_∂x / mlᵛx(η, I) * Δτ.V.x
    end
    @inbounds if checkbounds(Bool, q.y, I)
        ∂T_∂y = -∂ᵛy(T, I) / Δ.y
        q.y[I] += ∂T_∂y / mlᵛy(η, I) * Δτ.V.y
    end
    @inbounds if checkbounds(Bool, q.z, I)
        ∂T_∂z = -∂ᵛz(T, I) / Δ.z
        q.z[I] += ∂T_∂z / mlᵛz(η, I) * Δτ.V.z
    end
end
