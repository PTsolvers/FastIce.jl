using KernelAbstractions

using FastIce.GridOperators

@kernel function update_q!(q, T, λ_ρCp, Δτ, Δ)
    I = @index(Global, Cartesian)
    @inbounds if checkbounds(Bool, q.x, I)
        ∂T_∂x = λ_ρCp * ∂ᵛx(T, I) / Δ.x
        q.x[I] -= (q.x[I] + ∂T_∂x) * Δτ.q.x
    end
    @inbounds if checkbounds(Bool, q.y, I)
        ∂T_∂y = λ_ρCp * ∂ᵛy(T, I) / Δ.y
        q.y[I] -= (q.y[I] + ∂T_∂y) * Δτ.q.y
    end
    @inbounds if checkbounds(Bool, q.z, I)
        ∂T_∂z = λ_ρCp * ∂ᵛz(T, I) / Δ.z
        q.z[I] -= (q.z[I] + ∂T_∂z) * Δτ.q.z
    end
end

@kernel function update_T!(T, T_o, q, Δt, Δτ, Δ)
    I = @index(Global, Cartesian)
    @inbounds if checkbounds(Bool, T, I)
        ∂qx_∂x = ∂ᶜx(q.x, I) / Δ.x
        ∂qy_∂y = ∂ᶜy(q.y, I) / Δ.y
        ∂qz_∂z = ∂ᶜz(q.z, I) / Δ.z
        ∇q = ∂qx_∂x + ∂qy_∂y + ∂qz_∂z
        dTdt = (T[I] - T_o[I]) / Δt
        T[I] -= (dTdt + ∇q) * Δτ.T
    end
end