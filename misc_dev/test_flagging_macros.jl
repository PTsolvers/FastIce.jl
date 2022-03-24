using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@init_parallel_stencil(package=Threads, ndims=2)

include("../scripts/flagging_macros2D.jl")

macro all(A) esc(:($A[ix,iy])) end
macro inn(A) esc(:($A[ix+1,iy+1])) end

@enum MaterialID begin
    fluid; solid; gas
end

@parallel_indices (ix,iy) function update_A!(A,Ax,Ay,Axy,ϕ)
    @in_phase     ϕ fluid       @all(A)  =  1.0
    @in_phase     ϕ solid       @all(A)  = -1.0
    @in_phases_xi ϕ solid fluid @inn(Ax) =  2.0
    @in_phases_yi ϕ solid fluid @inn(Ay) = -2.0
    @in_phases_yi ϕ solid fluid @inn(Ay) = -2.0
    @in_phases_xy ϕ fluid fluid fluid fluid @all(Axy) = 100.0
    return
end

@views function main()
    ϕ           = Array{MaterialID}(undef,5,5)
    ϕ          .= fluid
    ϕ[1:3,1:2] .= solid 
    A           = Array{Float64}(undef,5,5)
    Ax          = @zeros(Float64,6,5)
    Ay          = @zeros(Float64,5,6)
    Axy         = @zeros(Float64,4,4)
    @parallel update_A!(A,Ax,Ay,Axy,ϕ)
    return A,Ax,Ay,Axy,ϕ
end

main()