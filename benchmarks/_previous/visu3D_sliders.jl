using GLMakie, UnPack
# mostly inspired from https://github.com/ntselepidis/FinalProjectRepo.jl/blob/main/scripts-part1/makie_volumeslice_example.jl

function generate_fig(inputs)
  @unpack ϕ, x3rot, y3rot, z3rot, x3, y3, z3, xc, yc, zc, R, lx, ly, lz, nx, ny, nz, sc = inputs

  # x = LinRange(0, x3rot[end,1,1]-x3rot[1,1,1], length(x3rot[:,1,1]))#1:size(ϕ)[1]
  # y = LinRange(0, y3rot[1,end,1]-y3rot[1,1,1], length(y3rot[1,:,1]))#1:size(ϕ)[2]
  # z = LinRange(0, z3rot[1,1,end]-z3rot[1,1,1], length(z3rot[1,1,:]))#1:size(ϕ)[3]./2

  x = LinRange(0, 5, size(ϕ)[1])
  y = LinRange(0, 5, size(ϕ)[2])
  z = LinRange(0, 1, size(ϕ)[3])

  fig = Figure()
  ax  = LScene(fig[1, 1], scenekw=(show_axis=false,))

  lsgrid = labelslidergrid!(
    fig,
    ["yz plane - x axis", "xz plane - y axis", "xy plane - z axis"],
    [1:length(x), 1:length(y), 1:length(z)]
    );
  fig[2, 1] = lsgrid.layout;

  plt = volumeslices!(ax, x, y, z, ϕ)

  # connect sliders to `volumeslices` update methods
  sl_yz, sl_xz, sl_xy = lsgrid.sliders

  on(sl_yz.value) do v; plt[:update_yz][](v) end
  on(sl_xz.value) do v; plt[:update_xz][](v) end
  on(sl_xy.value) do v; plt[:update_xy][](v) end

  set_close_to!(sl_yz, .5length(x))
  set_close_to!(sl_xz, .5length(y))
  set_close_to!(sl_xy, .5length(z))

  return fig
end
