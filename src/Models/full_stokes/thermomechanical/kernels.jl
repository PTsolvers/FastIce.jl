@kernel function update_qU!(qU, T, λ, Δ)
    ix, iy, iz = @index(Global, NTuple)
    @inbounds if checkbounds(Bool, qU.x, ix, iy, iz)
        @all(qU.x) = -λ * @∂_xi(T) / Δ.x
    end
    @inbounds if checkbounds(Bool, qU.y, ix, iy, iz)
        @all(qU.y) = -λ * @∂_yi(T) / Δ.y
    end
end

