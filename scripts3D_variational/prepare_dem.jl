using JLD2
using NetCDF

const GREENLAND_PATH = "../../Datasets/BedMachine/BedMachineGreenland-v5.nc"

function prepare_greenland()
    x = ncread(GREENLAND_PATH,"x")
    y = reverse(ncread(GREENLAND_PATH,"y"))
    @assert issorted(x)
    @assert issorted(y)
    x = LinRange(x[1],x[end],length(x))
    y = LinRange(y[1],y[end],length(y))
    bed     = reverse(ncread(GREENLAND_PATH,"bed")    ; dims=2)
    surface = reverse(ncread(GREENLAND_PATH,"surface"); dims=2)
    mask    = reverse(ncread(GREENLAND_PATH,"mask")   ; dims=2)
    jldsave("data/BedMachine/greenland.jld2";x,y,bed,surface,mask)
    return
end

prepare_greenland()