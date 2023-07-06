using CairoMakie
using JLD2
using Printf

nbump = 10
slope = -15.0

# ist    = 1

simdir = "out_visu/egu2023/nbump_$(nbump)_slope_$(slope)"
xc,xv,zc,zv,Ψ,wt,dem,mc = load("$simdir/static.jld2","xc","xv","zc","zv","Ψ","wt","dem","mc")    
Pr,τ,ε̇,ε̇II,V,T,ω,ηs     = load(@sprintf("%s/%04d.jld2",simdir,1),"Pr","τ","ε̇","ε̇II","V","T","ω","ηs")


fig = Figure(resolution=(800,500),fontsize=36)
# fig = Figure(resolution=(800,950),fontsize=36)
axs = (
    hmaps = (
        ε̇II = Axis(fig[1,1][1,1];aspect=DataAspect(),title=L"\log_{10}(\dot{\varepsilon}_{II})",xlabel=L"x",ylabel=L"z"),
        # T   = Axis(fig[2,1][1,1];aspect=DataAspect(),title=L"T",xlabel=L"x",ylabel=L"z"),
        # ω   = Axis(fig[3,1][1,1];aspect=DataAspect(),title=L"\omega",xlabel=L"x",ylabel=L"z"),
    ),
)

plts = (
    hmaps = (
        ε̇II = heatmap!(axs.hmaps.ε̇II,xc,zc,log10.(ε̇II);colormap=:turbo,colorrange=(-6,-2)),
        # T   = heatmap!(axs.hmaps.T  ,xc,zc,T  ;colormap=:magma,colorrange=(0.9,1)),
        # ω   = heatmap!(axs.hmaps.ω  ,xc,zc,ω  ;colormap=Reverse(:grays),colorrange=(0,0.06)),
    ),
)

for axname in eachindex(axs.hmaps)
    xlims!(axs.hmaps[axname],-0.5,0.5)
    ylims!(axs.hmaps[axname],zc[1],0.6)
end

mc_air = mc.ice[2:end-1]
push!(mc_air,Point(mc.ice[end][1],zv[end]+0.2))
push!(mc_air,Point(mc.ice[1  ][1],zv[end]+0.2))

plt_bed = [
    (
        bed = poly!(axs.hmaps[f],mc.bed;strokewidth=2,color=:gray),
        ice = poly!(axs.hmaps[f],mc_air;strokewidth=4,color=:white),
    ) for f in eachindex(axs.hmaps)
]

Colorbar(fig[1,1][1,2],plts.hmaps.ε̇II)
# Colorbar(fig[2,1][1,2],plts.hmaps.T  )
# Colorbar(fig[3,1][1,2],plts.hmaps.ω  )

display(fig)

record(fig,"video_$(nbump)_$(slope).mp4",1:50;framerate=5) do it
    if it == 1 return end
    local Pr,τ,ε̇,ε̇II,V,T,ω,ηs = load(@sprintf("%s/%04d.jld2",simdir,it),"Pr","τ","ε̇","ε̇II","V","T","ω","ηs")
    plts.hmaps.ε̇II[3] = log10.(ε̇II)
    # plts.hmaps.T[3]   = T
    # plts.hmaps.ω[3]   = ω
end

display(fig)


# fig = Figure(resolution=(1000,800),fontsize=32)
# axs = (
#     Vx  = Axis(fig[1,1];ylabel=L"z",xlabel=L"v_x"),
#     ε̇xy = Axis(fig[1,2];ylabel=L"z",xlabel=L"\dot{\varepsilon}_{xy}"),
# )

# for axname in eachindex(axs)
#     # xlims!(axs[axname],xc[1],xc[end])
#     ylims!(axs[axname],0.07,0.75)
# end

# xlims!(axs.ε̇xy,(-5e-3,nothing))

# exy = (V.x[ix,2:end] .- V.x[ix,1:end-1])./(zc[2]-zc[1])

# plts = (
#     Vx  = lines!(axs.Vx,V.x[ix,:].*100,zc;linewidth=2),
#     ε̇xy = lines!(axs.ε̇xy,exy,zv[2:end-1];linewidth=2),
# )

# display(fig)

# it = 50

# for slope in (-20.0,-15.0,-10.0,-5.0)
#     simdir = "out_visu/egu2023/nbump_$(nbump)_slope_$(slope)"
#     xc,xv,zc,zv,Ψ,wt,dem,mc = load("$simdir/static.jld2","xc","xv","zc","zv","Ψ","wt","dem","mc")    
#     Pr,τ,ε̇,ε̇II,V,T,ω,ηs     = load(@sprintf("%s/%04d.jld2",simdir,it),"Pr","τ","ε̇","ε̇II","V","T","ω","ηs")

#     nx,nz = length(xc),length(zc)
#     ix = nx ÷ 2
#     exy = (V.x[ix,2:end] .- V.x[ix,1:end-1])./(zc[2]-zc[1])
#     plts = (
#         Vx  = lines!(axs.Vx,V.x[ix,:],zc;linewidth=2),
#         ε̇xy = lines!(axs.ε̇xy,exy,zv[2:end-1];linewidth=2),
#     )
# end

# display(fig)