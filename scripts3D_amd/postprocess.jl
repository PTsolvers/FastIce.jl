using GLMakie, DelimitedFiles

function load_array(Aname,A)
    fname = string(Aname,".bin")
    fid=open(fname,"r"); read!(fid,A); close(fid)
end


function visualise(;outdir="out")
    (nx,ny,nz) = readdlm(joinpath(outdir,"nxyz.txt"),Int)
    xc   = zeros(Float32,nx);       load_array(joinpath(outdir,"out_xc"  ),xc)
    yc   = zeros(Float32,ny);       load_array(joinpath(outdir,"out_xc"  ),yc)
    zc   = zeros(Float32,nz);       load_array(joinpath(outdir,"out_xc"  ),zc)
    Pr   = zeros(Float32,nx,ny,nz); load_array(joinpath(outdir,"out_Pr"  ),Pr)
    Vmag = zeros(Float32,nx,ny,nz); load_array(joinpath(outdir,"out_Vmag"),Vmag)
    T    = zeros(Float32,nx,ny,nz); load_array(joinpath(outdir,"out_T"   ),T)
    # visualisation
    fig = Figure(resolution=(2000,800),fontsize=32)
    axs = (
    Pr   = Axis3(fig[1,1][1,1][1,1];aspect=:data,xlabel="x",ylabel="y",zlabel="z",title="Pr"),
    Vmag = Axis3(fig[1,1][1,2][1,1];aspect=:data,xlabel="x",ylabel="y",zlabel="z",title="|V|"),
    T    = Axis3(fig[1,1][1,3][1,1];aspect=:data,xlabel="x",ylabel="y",zlabel="z",title="T"),
    )
    plts = (
    Pr   = volumeslices!(axs.Pr  ,xc,yc,zc,Array(Pr  );colormap=:turbo),
    Vmag = volumeslices!(axs.Vmag,xc,yc,zc,Array(Vmag);colormap=:turbo),
    T    = volumeslices!(axs.T   ,xc,yc,zc,Array(T   );colormap=:turbo),
    )
    sgrid = SliderGrid(
    fig[2,1],
    (label = "yz plane - x axis", range = 1:length(xc)),
    (label = "xz plane - y axis", range = 1:length(yc)),
    (label = "xy plane - z axis", range = 1:length(zc)),
    )
    # connect sliders to `volumeslices` update methods
    sl_yz, sl_xz, sl_xy = sgrid.sliders
    on(sl_yz.value) do v; for prop in eachindex(plts) plts[prop][:update_yz][](v) end; end
    on(sl_xz.value) do v; for prop in eachindex(plts) plts[prop][:update_xz][](v) end; end
    on(sl_xy.value) do v; for prop in eachindex(plts) plts[prop][:update_xy][](v) end; end
    set_close_to!(sl_yz, .5length(xc))
    set_close_to!(sl_xz, .5length(yc))
    set_close_to!(sl_xy, .5length(zc))
    [Colorbar(fig[1,1][irow,icol][1,2],plts[(irow-1)*2+icol]) for irow in 1:1,icol in 1:3]
    display(fig)
    return
end

visualise(;outdir="out_8mh")
