using JuMP, Gurobi, CSV, DataFrames, Distributions, StatsBase, Random, Plots, COPT, GLPK

function estima_contrato(D::Matrix)
    S,N = size(D)
    p = ones(S) * 1/S

    model = Model(COPT.Optimizer)
    set_silent(model)

    @variable(model, T[1:3] ≥ 0) #valores limites do contrato e restricao A.25
    @variable(model, π) #premio de risco
    @variable(model, I[1:S]) #valor pago na idenizacao
    @variable(model, ϕ[1:S, 1:N]) # D - I
    @variable(model, v) #VaR usado no calculo do Cvar
    @variable(model, β[1:S] ≥ 0) #variavel do Cvar e restricao A.24
    @variable(model, δ1[1:S, 1:N])
    @variable(model, δ2[1:S, 1:N])
    @variable(model, δ3[1:S, 1:N])
    @variable(model, δ4[1:S])
    @variable(model, δ5[1:S])
    #@variable(model, y1[1:S, 1:N], COPT.COPT_WriteBin)
    #@variable(model, y2[1:S, 1:N], COPT.COPT_WriteBin)
    #@variable(model, y3[1:S], COPT.COPT_WriteBin)
    @variable(model, y1[1:S, 1:N], Bin)
    @variable(model, y2[1:S, 1:N], Bin)
    @variable(model, y3[1:S], Bin)
    @variable(model, z1[1:S, 1:N])
    @variable(model, z2[1:S, 1:N])
    @variable(model, z3[1:S, 1:N])
    @variable(model, z4[1:S])

    @objective(model, Min, (1 - λ)*sum(p[s] * (sum(D[s,n] for n in 1:N )  - I[s]) for s in 1:S) + λ*(v + sum(p[s]*β[s]/(1 - α) for s in 1:S) + π))

    @constraint(model, π ≥ (1 + τ)*sum(p[s]I[s] for s in 1:S)) #A.2 (premio maior do que a idenizacao media)
    @constraint(model, T[2] ≤ T[3]) #A.3
    @constraint(model, Ti ≤ T[1] ≤ Tf) #A.4
    @constraint(model, [s in 1:S, n in 1:N], D[s, n] == δ1[s,n] + δ2[s,n] + δ3[s,n]) #A.5
    @constraint(model, [s in 1:S, n in 1:N], ϕ[s,n] == δ1[s,n] + δ3[s, n]) #A.6
    @constraint(model, [s in 1:S, n in 1:N], z1[s,n] ≤ δ1[s, n]) #A.7.1
    @constraint(model, [s in 1:S, n in 1:N], δ1[s, n] ≤ T[1]) #A.7.2
    @constraint(model, [s in 1:S, n in 1:N], 0 ≤ z1[s,n]) #A.8.1
    @constraint(model, [s in 1:S, n in 1:N], z1[s,n] ≤ M[1]*y1[s, n]) #A.8.2
    @constraint(model, [s in 1:S, n in 1:N], 0 ≤ T[1] - z1[s, n]) #A.9.1
    @constraint(model, [s in 1:S, n in 1:N], T[1] - z1[s, n] ≤ M[1]*(1 - y1[s, n])) #A.9.2
    @constraint(model, [s in 1:S, n in 1:N], z3[s, n] ≤ δ2[s, n]) #A.10.1
    @constraint(model, [s in 1:S, n in 1:N], δ2[s, n] ≤ z2[s, n]) #A.10.2
    @constraint(model, [s in 1:S, n in 1:N], 0 ≤ z2[s, n]) #A.11.1
    @constraint(model, [s in 1:S, n in 1:N], z2[s, n] ≤ M[2]*y1[s, n]) #A.11.2
    @constraint(model, [s in 1:S, n in 1:N], 0 ≤ T[2] - z2[s, n]) #A.12.1
    @constraint(model, [s in 1:S, n in 1:N], T[2] - z2[s, n] ≤ M[2]*(1 - y1[s, n])) #A.12.2
    @constraint(model, [s in 1:S, n in 1:N], 0 ≤ z3[s, n]) #A.13.1
    @constraint(model, [s in 1:S, n in 1:N], z3[s, n] ≤ M[3]*y2[s, n]) #A.13.2
    @constraint(model, [s in 1:S, n in 1:N], 0 ≤ T[3] - z3[s, n]) #A.14.1
    @constraint(model, [s in 1:S, n in 1:N], T[3] - z3[s, n] ≤ M[3]*(1 - y2[s, n])) #A.14.2
    @constraint(model, [s in 1:S, n in 1:N], 0 ≤ δ3[s, n]) #A.15.1
    @constraint(model, [s in 1:S, n in 1:N], δ3[s, n] ≤ D[s,n]*y2[s, n]) #A.15.2
    @constraint(model, [s in 1:S, n in 1:N], y2[s, n] ≤ y1[s, n]) #A.16 y2 so ativa se y1 ta ativado
    @constraint(model, [s in 1:S], sum(D[s,n] - ϕ[s,n] for n in 1:N) == δ4[s] + δ5[s]) #A.17
    @constraint(model, [s in 1:S], I[s] == δ4[s]) #A.18
    @constraint(model, [s in 1:S], z4[s] ≤ δ4[s]) #A.19.1
    @constraint(model, [s in 1:S], δ4[s] ≤ T[3]) #A.19.2
    @constraint(model, [s in 1:S], 0 ≤ z4[s]) #A.20.1
    @constraint(model, [s in 1:S], z4[s] ≤ M[3]*y3[s]) #A.20.2
    @constraint(model, [s in 1:S], 0 ≤ T[3] - z4[s]) #A.21.1
    @constraint(model, [s in 1:S], T[3] - z4[s] ≤ M[3]*(1-y3[s])) #A.21.2
    @constraint(model, [s in 1:S], 0 ≤ δ5[s])# A.22.1
    @constraint(model, [s in 1:S], δ5[s] ≤ sum(D[s,n] for n in 1:N)*y3[s])# A.22.2
    @constraint(model, [s in 1:S], β[s] ≥ (sum(D[s,n] for n in 1:N) - I[s]) - v) #A.23

    optimize!(model)

    return value(π), value.(T), objective_value(model)
end

function MC_bootstrap(y, sample_size)
    x = ones(sample_size,1)

    StatsBase.sample!(y, x, replace=true, ordered=false)
    hist_x = histogram(y, label="Sampled", bins=:sqrt)
    hist_x = histogram!(x, label="Re-sampled MC", fillalpha=0.7, bins=:sqrt)
    return vec(x), hist_x
end

function QMC_LHS_empirical(y, sample_size)
    ran = rand(Uniform(0, 1), sample_size);
    s = zeros(sample_size,1)

    idx = randperm(sample_size)
    P = ((idx-ran[:,1])/sample_size).*100 # probability of the cdf
    s[:,1] = percentile(charges,P); # Trick for sampling
    hist_s = histogram(y, label="Sampled", bins=:sqrt)
    hist_s = histogram!(s, label="Re-sampled QMC", fillalpha=0.7, bins=:sqrt)
    return s, hist_s
end

Data = CSV.read("Databases/contracts.csv", DataFrame)

charges = Vector{Float64}(Data.charges)./1000
#charges = Matrix{Float64}(Data[!, ["charges"]])./1000


S = collect(100:100:1100)
Rep = 100

resultados = zeros(6, Rep)
resultados_opt = zeros(6, Rep)
resultados2 = zeros(6, Rep)

# Hyperparameters
N = 10
α = 0.95 #quantil do Cvar
λ = 0.5 #peso do cvar na função objetivo
τ = 0.3 #loading factor
Ti = 10
Tf = 150
M = [1e8, 1e8, 1e8]
p = 0.3

for r in 1:Rep
    D = QMC_LHS_empirical(charges, 500)[1]
    @time π0, T0 , z0 = estima_contrato(D)
    resultados[:,r] = vcat(p, π0, T0, z0)
end

for r in 1:Rep
    D = QMC_LHS_empirical(charges, 800)[1]
    @time π0, T0 , z0 = estima_contrato(D)
    resultados2[:,r] = vcat(p, π0, T0, z0)
end

for r in 1:Rep
    D = QMC_LHS_empirical(charges, 2000)[1]
    @time π0, T0 , z0 = estima_contrato(D)
    resultados_opt[:,r] = vcat(p, π0, T0, z0)
end

D_empirical = Matrix{Float64}(Data[!, ["charges"]])./1000
@time π0, T0 , z0 = estima_contrato(D_empirical)

D_opt = QMC_LHS_empirical(charges, 2000)[1]
@time π0_opt, T0_opt , z0_opt = estima_contrato(D_opt)

resultados3 = zeros(6, Rep)
resultados4 = zeros(6, Rep)
resultados5 = zeros(6, Rep)

for r in 1:Rep
    D = QMC_LHS_empirical(charges, 100)[1]
    @time π0, T0 , z0 = estima_contrato(D)
    resultados3[:,r] = vcat(p, π0, T0, z0)
end

for r in 1:Rep
    D = QMC_LHS_empirical(charges, 200)[1]
    @time π0, T0 , z0 = estima_contrato(D)
    resultados4[:,r] = vcat(p, π0, T0, z0)
end

for r in 1:Rep
    D = QMC_LHS_empirical(charges, 300)[1]
    @time π0, T0 , z0 = estima_contrato(D)
    resultados5[:,r] = vcat(p, π0, T0, z0)
end

resultados6 = zeros(6, Rep)
for r in 1:Rep
    D = QMC_LHS_empirical(charges, 10)[1]
    @time π0, T0 , z0 = estima_contrato(D)
    resultados6[:,r] = vcat(p, π0, T0, z0)
end

Plots.scatter(ones(Rep, 1)*500, resultados[6,:])
Plots.scatter!(ones(Rep, 1)*800, resultados2[6,:])
Plots.scatter!(ones(Rep, 1)*100, resultados3[6,:])
Plots.scatter!(ones(Rep, 1)*200, resultados4[6,:])
Plots.scatter!(ones(Rep, 1)*300, resultados5[6,:])
Plots.scatter!(ones(Rep, 1)*50, resultados6[6,:])
Plots.scatter!(ones(Rep, 1)*50, resultados6[6,:])
Plots.hline!([12.508336242883423])
a

Plots.scatter!(ones(Rep, 1)*2000, resultados_opt[6,:])

resultados_grid = Vector()
for i in 10:10:800
    D = QMC_LHS_empirical(charges, i)[1]
    @time π0, T0 , z0 = estima_contrato(D)
    resultados_grid = push!(resultados_grid, z0)
end

Plots.plot(collect(10:10:800), resultados_grid)
plot()

COPT.con



size(resultados)


π0, T0 , z0 = estima_contrato(D0)
