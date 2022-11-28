using Plots, Plots.Measures

default(fontfamily="Computer Modern", linewidth=4,  markershape=:circle, markersize=4,
        framestyle=:box, fillalpha=0.4, margin=5mm)
scalefontsizes(); scalefontsizes(1.3)

nprocs = (1, 8, 16, 32, 64, 128, 256)

T_eff  = (225.34, 223.05, 220.95, 219.35, 218.3, 217.4, 217.5)

T_effn = T_eff./T_eff[1]
nxyz   =  511^3

# ENV["GKSwstype"]="nul"

png(plot(collect(nprocs),collect(T_effn),
         xticks=(nprocs, string.(nprocs)), legend=false,
         xlabel="Number of AMD MI250x GPUs", ylabel="Parallel efficiency",
         dpi=150,size=(600, 380)),"weak_scale_lumi.png")
