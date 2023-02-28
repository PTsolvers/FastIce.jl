using JLD2

const GREENLAND_PATH = "../data/BedMachine/BedMachineGreenland-v5.nc"

function prepare_greenland()
    x = ncread(GREENLAND_PATH,"x")
    y = reverse(ncread(GREENLAND_PATH,"y"))
    @assert issorted(x)
    @assert issorted(y)
    x .-= x[1]; y .-= y[1]
    x = LinRange(x[1],x[end],length(x))
    y = LinRange(y[1],y[end],length(y))
    surface   = reverse(ncread(GREENLAND_PATH,"surface")  ; dims=2)
    thickness = reverse(ncread(GREENLAND_PATH,"thickness"); dims=2)
    bed       = reverse(ncread(GREENLAND_PATH,"bed")      ; dims=2)
    mask      = reverse(ncread(GREENLAND_PATH,"mask")     ; dims=2)
    
    return
end