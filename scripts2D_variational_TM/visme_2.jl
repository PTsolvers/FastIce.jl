using CairoMakie
using JLD2
using Printf

nbump = 20
slope = -20.0

fig = Figure(resolution=(1000,800),fontsize=38)
axs = (
    Vx  = Axis(fig[1,1];ylabel=L"z",xlabel=L"v_x\times 10^{4}"),
    ε̇xy = Axis(fig[1,2];ylabel=L"z",xlabel=L"\dot{\varepsilon}_{xy}"),
)

for axname in eachindex(axs)
    # xlims!(axs[axname],xc[1],xc[end])
    ylims!(axs[axname],0.07,0.75)
end

xlims!(axs.ε̇xy,(-5e-3,nothing))

it = 50

for slope in (-10.0,-19.0,-20.0)
    simdir = "out_visu/egu2023/nbump_$(nbump)_slope_$(slope)"
    xc,xv,zc,zv,Ψ,wt,dem,mc = load("$simdir/static.jld2","xc","xv","zc","zv","Ψ","wt","dem","mc")    
    Pr,τ,ε̇,ε̇II,V,T,ω,ηs     = load(@sprintf("%s/%04d.jld2",simdir,it),"Pr","τ","ε̇","ε̇II","V","T","ω","ηs")

    nx,nz = length(xc),length(zc)
    ix = nx ÷ 2
    exy = (V.x[ix,2:end] .- V.x[ix,1:end-1])./(zc[2]-zc[1])
    plts = (
        Vx  = lines!(axs.Vx,V.x[ix,:].*10000,zc;linewidth=3),
        ε̇xy = lines!(axs.ε̇xy,exy,zv[2:end-1];linewidth=3,label=L"\theta = %$(slope)^\circ"),
    )
end

axislegend(axs.ε̇xy)

display(fig)

save("comparison.png",fig)