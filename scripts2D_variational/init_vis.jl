@tiny function _kernel_init!(V, ε̇bg, xv, yv)
    ix, iy = @indices
    if ix ∈ axes(V.x, 1) && iy ∈ axes(V.x, 2)
        @inbounds V.x[ix, iy] = -xv[ix] * ε̇bg
    end
    if ix ∈ axes(V.y, 1) && iy ∈ axes(V.y, 2)
        @inbounds V.y[ix, iy] = yv[iy] * ε̇bg
    end
    return
end

@tiny function _kernel_update_vis_fields!(Vmag, Ψav, V, Ψ)
    ix, iy = @indices
    @inline isin(A) = checkbounds(Bool, A, ix, iy)
    @inbounds if isin(Ψ.not_air)
        pav = 0.0
        for idy = 1:2, idx = 1:2
            pav += Ψ.not_air[ix+idx, iy+idy]
        end
        Ψav.not_air[ix, iy] = pav / 8
    end
    # @inbounds if isin(Ψ.not_solid)
    #     pav = 0.0
    #     for idy = 1:2, idx = 1:2
    #         pav += Ψ.not_solid[ix+idx, iy+idy]
    #     end
    #     Ψav.not_solid[ix, iy] = pav / 8
    # end
    @inbounds if isin(Vmag)
        vxc = 0.5 * (V.x[ix+1, iy+1] + V.x[ix+2, iy+1])
        vyc = 0.5 * (V.y[ix+1, iy+1] + V.y[ix+1, iy+2])
        Vmag[ix, iy] = sqrt(vxc^2 + vyc^2)
    end
    return
end

const _init! = _kernel_init!(get_device())
const _update_vis! = _kernel_update_vis_fields!(get_device())

function init!(V, ε̇bg, xv, yv)
    wait(_init!(V, ε̇bg, xv, yv; ndrange=size(V.x) .+ (0, 1)))
    return
end

function update_vis!(Vmag, Ψav, V, Ψ)
    wait(_update_vis!(Vmag, Ψav, V, Ψ; ndrange=axes(Vmag)))
    return
end