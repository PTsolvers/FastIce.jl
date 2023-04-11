include("level_set_kernels.jl")

const _compute_levelset_from_polygon! = _kernel_compute_levelset_from_polygon!(get_device())

function compute_levelset!(op, Ψ, xv, yv, mc)
    wait(_compute_levelset_from_polygon!(op, Ψ, xv, yv, mc; ndrange=axes(Ψ)))
    return
end

compute_levelset!(Ψ, xv, yv, mc) = compute_levelset!(min, Ψ, xv, yv, mc)