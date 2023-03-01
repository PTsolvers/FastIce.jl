using TinyKernels
using Statistics
using StaticArrays
using CUDA
using TinyKernels.CUDABackend

@tiny function kernel_av8!(A_av,A)
    ix,iy,iz = @indices
    A_av[ix,iy,iz] = mean(A[ix+dix,iy+diy,iz+diz] for diz in 0:1, diy in 0:1, dix in 0:1)
    return
end

function main()
    nx,ny,nz = 5,6,7
    A    = CUDA.rand(Float64,nx,ny,nz)
    A_av = CuArray{Float64}(undef,nx-1,ny-1,nz-1)
    av8! = Kernel(kernel_av8!,CUDADevice())
    TinyKernels.device_synchronize(CUDADevice())
    wait(av8!(A_av,A;ndrange=axes(A_av)))
    display(A)
    display(A_av)
    return
end

main()
