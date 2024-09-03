@inline perturb(ψ) = abs(ψ) > oftype(ψ, 1e-20) ? ψ : (ψ > zero(ψ) ? oftype(ψ, 1e-20) : oftype(ψ, -1e-20))

# volume of 2- and 3-simplex
@inline vol(v::Vararg{Vec2{T},3}) where {T} = T(inv(2)) * abs(det([v[3] - v[1] v[2] - v[1]]))
@inline vol(v::Vararg{Vec3{T},4}) where {T} = T(inv(6)) * abs(det([v[2] - v[1] v[3] - v[1] v[4] - v[1]]))

# volume of 2-simplex with immersed boundary given by signed distances
function vol(ψ::NTuple{3,T}, v::NTuple{3,Vec2{T}}) where {T}
    if ψ[1] < 0 && ψ[2] < 0 && ψ[3] < 0 # ---
        return vol(v[1], v[2], v[3])
    elseif ψ[1] > 0 && ψ[2] > 0 && ψ[3] > 0 # +++
        return zero(T)
    end
    @inline vij(i, j) = v[j] * (ψ[i] / (ψ[i] - ψ[j])) - v[i] * (ψ[j] / (ψ[i] - ψ[j]))
    v12, v13, v23 = vij(1, 2), vij(1, 3), vij(2, 3)
    if ψ[1] < 0
        if ψ[2] < 0
            vol(v[1], v23, v13) + vol(v[1], v[2], v23)  # --+
        else
            if ψ[3] < 0
                vol(v[3], v12, v23) + vol(v[3], v[1], v12) # -+-
            else
                vol(v[1], v12, v13) # -++
            end
        end
    else
        if ψ[2] < 0
            if ψ[3] < 0
                vol(v[2], v13, v12) + vol(v[2], v[3], v13) # +--
            else
                vol(v12, v[2], v23) # +-+
            end
        else
            vol(v13, v23, v[3]) # ++-
        end
    end
end

# volume of 2d box with immersed boundary given by signed distances
function vol(ψ::NTuple{4,T}, Δ::NTuple{2,T}) where {T}
    v00, v01, v10, v11 = Vec2(zero(T), zero(T)), Vec2(Δ[1], zero(T)), Vec2(zero(T), Δ[2]), Vec2(Δ[1], Δ[2])
    ψ00, ψ01, ψ10, ψ11 = perturb.(ψ)
    return vol((ψ00, ψ01, ψ11), (v00, v01, v11)) +
           vol((ψ00, ψ11, ψ10), (v00, v11, v10))
end

# volume of 3-simplex with immersed boundary given by signed distances
function vol(ψ::NTuple{4,T}, v::NTuple{4,Vec3{T}}) where {T}
    @inline vij(i, j) = v[j] * (ψ[i] / (ψ[i] - ψ[j])) - v[i] * (ψ[j] / (ψ[i] - ψ[j]))
    nneg = count(ψ .< 0)
    if nneg == 0     # ++++
        return zero(T)
    elseif nneg == 1 # -+++
        if ψ[1] < 0
            return vol(v[1], vij(1, 2), vij(1, 3), vij(1, 4))
        elseif ψ[2] < 0
            return vol(v[2], vij(2, 1), vij(2, 3), vij(2, 4))
        elseif ψ[3] < 0
            return vol(v[3], vij(3, 1), vij(3, 2), vij(3, 4))
        else # ψ[4] < 0
            return vol(v[4], vij(4, 1), vij(4, 2), vij(4, 3))
        end
    elseif nneg == 2 # --++
        if ψ[1] < 0 && ψ[2] < 0
            return vol(v[1], v[2], vij(1, 3), vij(2, 4)) +
                   vol(vij(2, 3), v[2], vij(1, 3), vij(2, 4)) +
                   vol(v[1], vij(1, 4), vij(1, 3), vij(2, 4))
        elseif ψ[1] < 0 && ψ[3] < 0
            return vol(v[1], v[3], vij(1, 4), vij(3, 2)) +
                   vol(vij(3, 4), v[3], vij(1, 4), vij(3, 2)) +
                   vol(v[1], vij(1, 2), vij(1, 4), vij(3, 2))
        elseif ψ[1] < 0 && ψ[4] < 0
            return vol(v[1], v[4], vij(1, 2), vij(4, 3)) +
                   vol(vij(4, 2), v[4], vij(1, 2), vij(4, 3)) +
                   vol(v[1], vij(1, 3), vij(1, 2), vij(4, 3))
        elseif ψ[2] < 0 && ψ[3] < 0
            return vol(v[3], v[2], vij(3, 1), vij(2, 4)) +
                   vol(vij(2, 1), v[2], vij(3, 1), vij(2, 4)) +
                   vol(v[3], vij(3, 4), vij(3, 1), vij(2, 4))
        elseif ψ[2] < 0 && ψ[4] < 0
            return vol(v[4], v[2], vij(4, 1), vij(2, 3)) +
                   vol(vij(2, 1), v[2], vij(4, 1), vij(2, 3)) +
                   vol(v[4], vij(4, 3), vij(4, 1), vij(2, 3))
        else # ψ[3] < 0 && ψ[4] < 0
            return vol(v[3], v[4], vij(3, 1), vij(4, 2)) +
                   vol(vij(4, 1), v[4], vij(3, 1), vij(4, 2)) +
                   vol(v[3], vij(3, 2), vij(3, 1), vij(4, 2))
        end
    elseif nneg == 3 # ---+
        vol_tot = vol(v[1], v[2], v[3], v[4])
        if ψ[1] >= 0
            return vol_tot - vol(v[1], vij(1, 2), vij(1, 3), vij(1, 4))
        elseif ψ[2] >= 0
            return vol_tot - vol(v[2], vij(2, 1), vij(2, 3), vij(2, 4))
        elseif ψ[3] >= 0
            return vol_tot - vol(v[3], vij(3, 1), vij(3, 2), vij(3, 4))
        else # ψ[4] >= 0
            return vol_tot - vol(v[4], vij(4, 1), vij(4, 2), vij(4, 3))
        end
    else # ----
        return vol(v[1], v[2], v[3], v[4])
    end
end

# volume of 3d-box with immersed boundary given by signed distances
function vol(ψ::NTuple{8,T}, Δ::NTuple{3,T}) where {T}
    v000, v001, v010, v011 = Vec3(zero(T), zero(T), zero(T)), Vec3(Δ[1], zero(T), zero(T)), Vec3(zero(T), Δ[2], zero(T)), Vec3(Δ[1], Δ[2], zero(T))
    v100, v101, v110, v111 = Vec3(zero(T), zero(T), Δ[3]), Vec3(Δ[1], zero(T), Δ[3]), Vec3(zero(T), Δ[2], Δ[3]), Vec3(Δ[1], Δ[2], Δ[3])
    ψ = perturb.(ψ)
    return vol((ψ[1], ψ[5], ψ[3], ψ[2]), (v000, v100, v010, v001)) +
           vol((ψ[7], ψ[5], ψ[3], ψ[8]), (v110, v100, v010, v111)) +
           vol((ψ[6], ψ[5], ψ[8], ψ[2]), (v101, v100, v111, v001)) +
           vol((ψ[4], ψ[8], ψ[3], ψ[2]), (v011, v111, v010, v001)) +
           vol((ψ[8], ψ[5], ψ[3], ψ[2]), (v111, v100, v010, v001))
end

# volume fraction inside a 2D grid cell
function ψ2ω(ψ::NTuple{2,NTuple{2,T}}, Δ::NTuple{2,T}) where {T}
    ψf = (ψ[1][1], ψ[1][2], ψ[2][1], ψ[2][2])
    return vol(ψf, Δ) / prod(Δ)
end

# volume fraction inside a 3D grid cell
function ψ2ω(ψ::NTuple{2,NTuple{2,NTuple{2,T}}}, Δ::NTuple{3,T}) where {T}
    ψf = (ψ[1][1][1], ψ[1][1][2], ψ[1][2][1], ψ[1][2][2], ψ[2][1][1], ψ[2][1][2], ψ[2][2][1], ψ[2][2][2])
    return vol(ψf, Δ) / prod(Δ)
end
