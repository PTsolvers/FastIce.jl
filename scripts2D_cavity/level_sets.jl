include("level_set_kernels.jl")

const _compute_levelset_from_polygon! = _kernel_compute_levelset_from_polygon!(get_device())

function compute_levelset!(op, Ψ, xv, yv, mc)
    wait(_compute_levelset_from_polygon!(op, Ψ, xv, yv, mc; ndrange=axes(Ψ)))
    return
end

# by default, compute union of new and current levelset
compute_levelset!(Ψ, xv, yv, mc) = compute_levelset!(min, Ψ, xv, yv, mc)

const _extrapolate_with_levelset! = _kernel_extrapolate_with_levelset!(get_device())

function extrapolate_with_levelset!(∂A_∂τ, A, Ψ, Δx, Δy)
    wait(_extrapolate_with_levelset!(∂A_∂τ, A, Ψ, Δx, Δy; ndrange=(axes(A, 1)[2:end-1], axes(A, 2)[2:end-1])))
    return
end