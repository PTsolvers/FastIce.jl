using Plots, Plots.Measures

default(fontfamily="Computer Modern", linewidth=4,  markershape=:circle, markersize=4,
        framestyle=:box, fillalpha=0.4, margin=5mm)
scalefontsizes(); scalefontsizes(1.3)

nz = (63, 127, 255, 511)

T_init  = (0.294776409,0.605874031,3.123633533,43.06257976)

png(plot(collect(nz),collect(T_init),
         xticks=(nz, string.(nz)), legend=false,
         xlabel="Number of grid points in z direction", ylabel="Initialisation time",
         dpi=150,size=(600, 380)),"init_strong_scale_lumi.png")
