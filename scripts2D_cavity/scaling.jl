ly       = 30.0 # m
A0       = 1e-23 # Pa s ^ m ~1e-24
ρg0      = 910 * 9.81 # m / s ^ 2
# nondim
npow     = 3.0
mpow     = (1 - npow) / npow
# scales
l_sc     = ly
τ_sc     = ρg0 * l_sc              # buoyancy
t_sc     = (A0 / τ_sc^(-npow))^-1  # buoyancy
t_sc2     = A0^-1 / τ_sc^npow  # buoyancy
# τ_sc     = A0^(-1/n) * ε̇bg # shear
# t_sc     = 1 / ε̇bg         # shear
η_sc     = τ_sc * t_sc
v_sc     = l_sc / t_sc

P_m   = (-0.0, 2.0)
V_m   = 0.15
τII_m = 0.9
ϵII_m = 0.9

P_s   = P_m .* τ_sc ./ 1e6
V_s   = V_m * v_sc * 86400 * 100 # cm / day
τII_s = τII_m * τ_sc / 1e6 # MPa
ϵII_s = ϵII_m / t_sc * 86400
