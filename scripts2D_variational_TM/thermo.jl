include("thermo_kernels.jl")

const _update_qT! = _kernel_update_qT!(get_device())
const _update_ρU! = _kernel_update_ρU!(get_device())

function update_qT!(qT,T,wt,λ,T_atm,dx,dy)
    qT_inn = (x=inn_x(qT.x), y=inn_y(qT.y))
    vf_inn(vf) = (
        c = vf.c,
        x = inn_x(vf.x),
        y = inn_y(vf.y),
    )
    wt_inn = (
        not_air   = vf_inn(wt.not_air  ),
        not_solid = vf_inn(wt.not_solid),
    )
    wait(_update_qT!(qT_inn,T,wt_inn,λ,T_atm,dx,dy; ndrange=axes(T)))
    return
end

function update_ρU!(ρU,qT,τ,ε̇,wt,ρU_atm,dt,dx,dy)
    wait(_update_ρU!(ρU,qT,τ,ε̇,wt,ρU_atm,dt,dx,dy; ndrange=axes(ρU)))
    return
end