@inline perturb(ϕ) = abs(ϕ) > 1e-20 ? ϕ : (ϕ > 0 ? 1e-20 : -1e-20)

@inline trivol(v1, v2, v3) = 0.5 * abs(cross(v3 - v1, v2 - v1))

function volfrac(tri, ϕ::Vec3{T})::T where {T}
    v1, v2, v3 = tri
    if ϕ[1] < 0 && ϕ[2] < 0 && ϕ[3] < 0 # ---
        return trivol(v1, v2, v3)
    elseif ϕ[1] > 0 && ϕ[2] > 0 && ϕ[3] > 0 # +++
        return 0.0
    end
    @inline vij(i, j) = tri[j] * (ϕ[i] / (ϕ[i] - ϕ[j])) - tri[i] * (ϕ[j] / (ϕ[i] - ϕ[j]))
    v12, v13, v23 = vij(1, 2), vij(1, 3), vij(2, 3)
    if ϕ[1] < 0
        if ϕ[2] < 0
            trivol(v1, v23, v13) + trivol(v1, v2, v23)  # --+
        else
            if ϕ[3] < 0
                trivol(v3, v12, v23) + trivol(v3, v1, v12) # -+-
            else
                trivol(v1, v12, v13) # -++
            end
        end
    else
        if ϕ[2] < 0
            if ϕ[3] < 0
                trivol(v2, v13, v12) + trivol(v2, v3, v13) # +--
            else
                trivol(v12, v2, v23) # +-+
            end
        else
            trivol(v13, v23, v3) # ++-
        end
    end
end

function volfrac(rect::Rect2{T}, ϕ::Vec4{T}) where {T}
    or, ws = origin(rect), widths(rect)
    v1, v2, v3, v4 = or, or + Vec(ws[1], 0.0), or + ws, or + Vec(0.0, ws[2])
    ϕ1, ϕ2, ϕ3, ϕ4 = perturb.(ϕ)
    return volfrac(Vec(v1, v2, v3), Vec3{T}(ϕ1, ϕ2, ϕ3)) +
           volfrac(Vec(v1, v3, v4), Vec3{T}(ϕ1, ϕ3, ϕ4))
end

include("volume_fraction_kernels.jl")

const _compute_volume_fractions_from_level_set! = _kernel_compute_volume_fractions_from_level_set!(get_device())

function compute_volume_fractions_from_level_set!(wt, Ψ, dx, dy)
    wt_inn = (; c=wt.c, x=inn_x(wt.x), y=inn_y(wt.y), xy=wt.xy)
    wait(_compute_volume_fractions_from_level_set!(wt_inn, Ψ, dx, dy; ndrange=axes(wt.c)))
    bc_x_neumann!(0.0, wt.x)
    bc_y_neumann!(0.0, wt.y)
    return
end