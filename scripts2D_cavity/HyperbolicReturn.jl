using Plots, SpecialFunctions

function ReturnMappingHyperbolicPlusVonMises()

    println("Grrr")

    C   = 5e7
    σt  = C/2
    σc  = 3e8
    ϕ   = 30.0*pi/180
    d0  = 1.0
    σd  = C/2
    Pd  = C
    σ_vm = 200e6
    
    P   = -5e7:1e6:5e8
    τ   = 0:1e6:4e8
    c   = ones(size(P))
    c   = 0.5*erfc.( ( (P.-Pd)./σd) )
    d   = 0.5*erfc.( ( (P.-σc)./(2*σd)) )

    Fhy   = zeros(length(P), length(τ))
    Qhy   = zeros(length(P), length(τ))
    Fdp   = zeros(length(P), length(τ))
    ψ     = zeros(length(P), length(τ))

    for i=1:length(P)
        for j=1:length(τ)
            dQdP   = -c[i]*sin(ϕ)
            dQdτ   = τ[j] ./ sqrt.( τ[j].^2 .+ (C*cos(ϕ)-σt*sin(ϕ)).^2 )
            ψ[i,j] = atand( -(dQdP./dQdτ) )
            Fhy[i,j] = sqrt( τ[j]^2 + (C*cos(ϕ)-σt*sin(ϕ))^2 ) - (d[i]*(C*cos(ϕ)+P[i]*sin(ϕ)) + (1-d[i]).*σ_vm) 
            Qhy[i,j] = sqrt( τ[j]^2 + (C*cos(ϕ)-σt*sin(ϕ))^2 ) - (C*cos(ϕ)+c[i]*P[i]*sin(ϕ))
            Fdp[i,j] = τ[j] - C*cos(ϕ) - P[i]*sin(ϕ)
        end
    end

    Ptrial = [ -3e7 -3e7 -2.7e7 1.5e7 6.5e7 9.00e7 3.2e8 4e8]
    τtrial = [0.2e7  1e7  2.5e7 6.5e7   9e7 9.75e7 2.1e8 2.1e8]
    ηve    = 1e20  
    Kve    = 2e20

     # Here all derivatives are difined in terms of trial state
     c      = 0.5*erfc.( ( (Ptrial.-Pd)./σd) )
     d      = 0.5*erfc.( ( (Ptrial.-σc)./(2*σd)) )
     @show  FHy    = sqrt.( τtrial.^2 .+ (C*cos(ϕ)-σt*sin(ϕ)).^2 ) .- (C*cos(ϕ).+Ptrial*sin(ϕ))
     λ̇      = (FHy.>0) .* FHy ./ (ηve .+ Kve.*sin(ϕ)^2 .*c)
     dcdP   = .-exp.(.-(Ptrial.-Pd).^2 ./σd.^2) / (sqrt(π)*σd)
     dQdP   = -c.*sin(ϕ) 
     dQdτ   = τtrial ./ sqrt.( τtrial.^2 .+ (C*cos(ϕ)-σt*sin(ϕ)).^2 )
     # Newton
     niter  = 10
     r      = zeros(niter, length(c))
     iter   = 0
     PHy    = Ptrial 
     τHy    = τtrial 
     while iter<niter
         iter  += 1
         PHy    = Ptrial .- λ̇.*Kve.*dQdP
         τHy    = τtrial .- λ̇.*ηve.*dQdτ
         @show  FHy    = sqrt.( τHy.^2 .+ (C*cos(ϕ).-σt*sin(ϕ)).^2 ) .- (d.*(C.*cos(ϕ).+PHy.*sin(ϕ)) .+ (1 .-d).*σ_vm)
         dFdλ̇   = d*Kve.*dQdP.*sin(ϕ) .- dQdτ.*ηve.*τHy ./ sqrt.( (C.*cos(ϕ).-σt.*sin(ϕ)).^2 .+ τHy.^2 )
         λ̇    .-= FHy./dFdλ̇
         r[iter,:] .= FHy[:]
     end

    p1 = plot()
    p1 = heatmap!(P, τ, Fhy'./1e6)
    p1 = contour!(P, τ, Fhy'./1e6, levels=[0,0], c=:white)
    p1 = contour!(P, τ, Qhy'./1e6, levels=[0,0], c=:black, linestyle=:dash)
    p1 = contour!(P, τ, Fdp'./1e6, levels=[0,0], c=:white)
    p1 = plot!(Ptrial, τtrial, markershape=:star, c=:white, label=:none)
    p1 = plot!(PHy, τHy, markershape=:xcross, c=:black, label=:none)
    for i=1:length(Ptrial)
        x = [Ptrial[i]; PHy[i]]
        y = [τtrial[i]; τHy[i]]
        p1 = plot!(x, y, label=:none, c=:white)
    end
    p2 = plot()
    for i=1:length(Ptrial)
        p2 = plot!(1:iter, log10.( abs.(r[1:iter,i] .+ 1e-13)), markershape=:circle, linewidth=1.0)
    end
    display( plot(p1, p2))
    @show r

end

function ReturnMappingHyperbolic()

    println("Grrr")

    C   = 5e7
    σt  = C/2
    ϕ   = 30.0*pi/180
    d0  = 1.0
    σd  = C/2
    Pd  = C
    
    P   = -5e7:1e6:1e8
    τ   = 0:1e6:1e8
    c   = ones(size(P))
    c   = 0.5*erfc.( ( (P.-Pd)./σd) )

    Fhy   = zeros(length(P), length(τ))
    Qhy   = zeros(length(P), length(τ))
    Fdp   = zeros(length(P), length(τ))
    ψ     = zeros(length(P), length(τ))

    for i=1:length(P)
        for j=1:length(τ)
            dQdP   = -c[i]*sin(ϕ)
            dQdτ   = τ[j] ./ sqrt.( τ[j].^2 .+ (C*cos(ϕ)-σt*sin(ϕ)).^2 )
            ψ[i,j] = atand( -(dQdP./dQdτ) )
            Fhy[i,j] = sqrt( τ[j]^2 + (C*cos(ϕ)-σt*sin(ϕ))^2 ) - (C*cos(ϕ)+P[i]*sin(ϕ))
            Qhy[i,j] = sqrt( τ[j]^2 + (C*cos(ϕ)-σt*sin(ϕ))^2 ) - (C*cos(ϕ)+c[i]*P[i]*sin(ϕ))
            Fdp[i,j] = τ[j] - C*cos(ϕ) - P[i]*sin(ϕ)
        end
    end

    # Ptrial = [6.5e7]
    # τtrial = [9e7]
    # Ptrial = [-2.7e7]
    # τtrial = [2.5e7]
    # Ptrial = [-3e7]
    # τtrial = [1e7]
    Ptrial = [ -3e7 -3e7 -2.7e7 1.5e7 6.5e7 9.00e7]
    τtrial = [0.2e7  1e7  2.5e7 6.5e7   9e7 9.75e7]
    ηve    = 1e20  
    Kve    = 2e20

    # Return mapping to DP
    @show FDp    = τtrial .- C*cos(ϕ) .- Ptrial*sin(ϕ)
    λ̇      = (FDp.>0) .* FDp ./ ηve
    PDP    = Ptrial
    τDP    = τtrial .- λ̇*ηve 
    @show FDp      = τDP .- C*cos(ϕ) .- PDP*sin(ϕ)

    # Return mapping to hyperbolic yield
    @show  FHy    = sqrt.( τtrial.^2 .+ (C*cos(ϕ)-σt*sin(ϕ)).^2 ) .- (C*cos(ϕ).+Ptrial*sin(ϕ))
    c      = 0.5*erfc.( ( (Ptrial.-Pd)./σd) )
    λ̇      = (FHy.>0) .* FHy ./ (ηve .+ Kve.*sin(ϕ)^2 .*c)
    PHy    = Ptrial
    τHy    = τtrial
    # Brute force
    iter   = 0
    niter  = 20
    r      = zeros(niter, length(c))
    # while iter<niter
    #     iter += 1
    #     @show c      = 0.5*erfc.( ( (PHy.-Pd)./σd) )
    #     dcdP   = .-exp.(.-(PHy.-Pd).^2 ./σd.^2) / (sqrt(π)*σd)
    #     dQdP   = -c.*sin(ϕ) .- 0*PHy.*dcdP
    #     dQdτ   = τHy ./ sqrt.( τHy.^2 .+ (C*cos(ϕ)-σt*sin(ϕ)).^2 )
    #     PHy    = Ptrial .- λ̇*Kve.*dQdP
    #     τHy    = τtrial .- λ̇*ηve.*dQdτ
    #     @show  FHy    = sqrt.( τHy.^2 .+ (C*cos(ϕ).-σt*sin(ϕ)).^2 ) .- (C*cos(ϕ).+PHy*sin(ϕ))
    #     λ̇     .+= 1/ηve .* FHy
    #     r[iter,:] .= FHy[:]
    # end

    # # Here all derivatives are difined in terms of trial state
    # c      = 0.5*erfc.( ( (Ptrial.-Pd)./σd) )
    # dcdP   = .-exp.(.-(Ptrial.-Pd).^2 ./σd.^2) / (sqrt(π)*σd)
    # dQdP   = -c.*sin(ϕ) 
    # dQdτ   = τtrial ./ sqrt.( τtrial.^2 .+ (C*cos(ϕ)-σt*sin(ϕ)).^2 )
    # # Brute force
    # iter   = 0
    # niter  = 20
    # r      = zeros(niter, length(c))
    # while iter<niter
    #     iter += 1
    #     PHy    = Ptrial .- λ̇*Kve.*dQdP
    #     τHy    = τtrial .- λ̇*ηve.*dQdτ
    #     @show  FHy    = sqrt.( τHy.^2 .+ (C*cos(ϕ).-σt*sin(ϕ)).^2 ) .- (C*cos(ϕ).+PHy*sin(ϕ))
    #     λ̇     .+= 1/ηve .* FHy
    #     r[iter,:] .= FHy[:]
    # end

    # Here all derivatives are difined in terms of trial state
    c      = 0.5*erfc.( ( (Ptrial.-Pd)./σd) )
    dcdP   = .-exp.(.-(Ptrial.-Pd).^2 ./σd.^2) / (sqrt(π)*σd)
    dQdP   = -c.*sin(ϕ) 
    dQdτ   = τtrial ./ sqrt.( τtrial.^2 .+ (C*cos(ϕ)-σt*sin(ϕ)).^2 )
    # Newton
    niter  = 5
    r      = zeros(niter, length(c))
    iter   = 0
    while iter<niter
        iter  += 1
        PHy    = Ptrial .- λ̇.*Kve.*dQdP
        τHy    = τtrial .- λ̇.*ηve.*dQdτ
        @show  FHy    = sqrt.( τHy.^2 .+ (C*cos(ϕ)-σt*sin(ϕ)).^2 ) .- (C*cos(ϕ).+PHy*sin(ϕ))
        dFdλ̇   = Kve.*dQdP.*sin(ϕ) .- dQdτ.*ηve.*τHy ./ sqrt.( (C.*cos(ϕ).-σt.*sin(ϕ)).^2 .+ τHy.^2 )
        λ̇    .-= FHy./dFdλ̇
        r[iter,:] .= FHy[:]
    end

    p1 = plot()
    p1 = heatmap!(P, τ, Fhy'./1e6)
    p1 = contour!(P, τ, Fhy'./1e6, levels=[0,0], c=:white)
    p1 = contour!(P, τ, Qhy'./1e6, levels=[0,0], c=:black, linestyle=:dash)
    p1 = contour!(P, τ, Fdp'./1e6, levels=[0,0], c=:white)
    p1 = plot!(Ptrial, τtrial, markershape=:star, c=:white, label=:none)
    p1 = plot!(PDP, τDP, markershape=:cross, c=:black, label=:none)
    p1 = plot!(PHy, τHy, markershape=:xcross, c=:black, label=:none)
    for i=1:length(Ptrial)
        x = [Ptrial[i]; PHy[i]]
        y = [τtrial[i]; τHy[i]]
        p1 = plot!(x, y, label=:none, c=:white)
    end
    p2 = plot()
    for i=1:length(Ptrial)
        p2 = plot!(1:iter, log10.( abs.(r[1:iter,i] .+ 1e-13)), markershape=:circle, linewidth=1.0)
    end
    # p2 = plot(P, c)
    p3 = heatmap(P, τ, ψ')
    # display(plot(p1, p2, p3, layout=(3,1)))   
    display(plot(p1, p2, p3, layout=(3,1)))

end


# ReturnMappingHyperbolicPlusVonMises()

ReturnMappingHyperbolic()