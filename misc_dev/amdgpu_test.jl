using AMDGPU

@inbounds function memcopy_triad!(A, B, C, s, side)
    ix = (workgroupIdx().x - 1) * workgroupDim().x + workitemIdx().x
    iy = (workgroupIdx().y - 1) * workgroupDim().y + workitemIdx().y
    if side == 1 && ix < size(A,1) / 2
        A[ix,iy] = B[ix,iy] + s*C[ix,iy]
    elseif side == 2 && ix >= size(A,1) / 2
        A[ix,iy] = B[ix,iy] + 2s*C[ix,iy]
    end
    return
end

function main()
    println("AMDGPU functional: $(AMDGPU.functional())")
    
    nx, ny  = 10, 12
    threads = (32, 8)
    grid    = (nx, ny)

    A = AMDGPU.zeros(Float64,nx,ny)
    B = AMDGPU.ones(Float64,nx,ny)
    C = AMDGPU.ones(Float64,nx,ny)

    s = 1.0

    AMDGPU.set_default_agent!(AMDGPU.get_agents(:gpu)[1])
    qs = Vector{AMDGPU.HSAQueue}(undef,2)
    for iside = 1:2
        qs[iside] = AMDGPU.HSAQueue(get_default_agent())
        priority = iside == 1 ? AMDGPU.HSA.QUEUE_PRIORITY_LOW : AMDGPU.HSA.QUEUE_PRIORITY_HIGH
        AMDGPU.HSA.amd_queue_set_priority(qs[iside].queue,priority)
    end

    signals = Vector{AMDGPU.RuntimeEvent{AMDGPU.HSAStatusSignal}}(undef,2)

    for side = 1:2
        signals[side] = @roc groupsize=threads gridsize=grid queue=qs[side] memcopy_triad!(A, B, C, s, side)
    end

    for side = 1:2
        wait(signals[side])
    end

    println("Done")

    return A
end

main()