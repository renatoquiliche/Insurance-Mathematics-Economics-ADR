using Random, Distributions, JuMP, Gurobi, XLSX, DataFrames, FreqTables, StatsPlots, HypothesisTests, CSV
using BenchmarkTools

function estima_contrato(D::Matrix)
    S,N = size(D)
    p = ones(S) * 1/S

    model = Model(Gurobi.Optimizer)
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

function gera_cenarios(base::Matrix, S::Int64, N::Int64, p::Float64)
    Random.seed!(0);
    D = zeros(S, N)
    cluster = zeros(S)
    for s in 1:S
        cluster[s] = rand(base[:, 2], 1)[1]
        ind = findall(i -> i==cluster[s], base[:, 2])
        for n in 1:N
            w = rand(Bernoulli(p))
            if w
                D[s, n] = rand(base[ind, 1],1)[1]
            end
        end
    end
    return D, cluster
end

function calcula_receita(D::Matrix, cluster::Vector, dict::Dict, mix::Bool)

    S,N = size(D)

    receita = zeros(S)
    indenizacao_aux = zeros(S, N)
    indenizacao = zeros(S)

    for s in 1:S
        for n in 1:N
            if mix
                indenizacao_aux[s, n] = min(D_teste[s, n], dict[4.0][2])
            else
                indenizacao_aux[s, n] = min(D_teste[s, n], dict[cluster[s]][2])
            end
        end
        if mix
            indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict[4.0][3])
            receita[s] = dict[4.0][4] .- indenizacao[s]
        else
            indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict[cluster[s]][3])
            receita[s] = dict[cluster[s]][4] .- indenizacao[s]
        end
    end

    return receita, indenizacao
end

cd("G:\\My Drive\\PUC-Rio\\PUC-Rio 2022-1\\Trabalho final ADR - Bruno Fanzeres\\Script")

#We create a random seed


##
############################################
# Analise da convergencia "s" ##
############################################

base = XLSX.readdata("Database.xlsx", "Sheet1!B2:N1339")
dados = Matrix{Float64}(base[:, [4, end]])

"""
Python code
"""
using PyCall

ENV["PYTHON"] = "C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
Pkg.build("PyCall")

pd = pyimport("pandas")
np = pyimport("numpy")
scipystats = pyimport("scipy.stats")

f_x = scipystats.randint(1,7)

"""
end
"""

ind0 = findall(i -> i==0.0, dados[:, 2])
ind1 = findall(i -> i==1.0, dados[:, 2])
ind2 = findall(i -> i==2.0, dados[:, 2])
ind3 = findall(i -> i==3.0, dados[:, 2])

N = 10
α = 0.95 #quantil do Cvar
λ = 0.5 #peso do cvar na função objetivo
τ = 0.3 #loading factor
Ti = 10
Tf = 150
M = [1e8, 1e8, 1e8]
p = 0.3

S = collect(100:100:1100)
Rep = 10
#resultados = zeros(6, length(S), Rep)
resultados0 = zeros(6, length(S), Rep)
resultados1 = zeros(6, length(S), Rep)
resultados3 = zeros(6, length(S), Rep)
resultados2 = zeros(6, length(S), Rep)

for i in 1:length(S)
    for r in 1:Rep
        #D, cluster = gera_cenarios(dados, S[i], N, p)
        print("Model: ", "Cluster 0 ", "Replication: ", r, " Scenarios: ", S[i], " ")
        D0, cluster0 = gera_cenarios(dados[ind0,:], S[i], N, p)
        #π, T , z = estima_contrato(D)
        @time π0, T0 , z0 = estima_contrato(D0)
        #resultados[:,i,r]  = vcat(p, π, T, z)
        resultados0[:,i,r] = vcat(p, π0, T0, z0)
    end
end

for i in 1:length(S)
    for r in 1:Rep
        print("Model: ", "Cluster 1 ", "Replication: ", r, " Scenarios: ", S[i], " ")
        D1, cluster1 = gera_cenarios(dados[ind1,:], S[i], N, p)
        @time π1, T1 , z1 = estima_contrato(D1)
        resultados1[:,i,r] = vcat(p, π1, T1, z1)
    end
end

for i in 1:length(S)
    for r in 1:Rep
        print("Model: ", "Cluster 2 ", "Replication: ", r, " Scenarios: ", S[i], " ")
        D2, cluster2 = gera_cenarios(dados[ind2,:], S[i], N, p)
        @time π2, T2 , z2 = estima_contrato(D2)
        resultados2[:,i,r] = vcat(p, π2, T2, z2)
    end
end

for i in 1:length(S)
    for r in 1:Rep
        print("Model: ", "Cluster 3 ", "Replication: ", r, " Scenarios: ", S[i], " ")
        D3, cluster3 = gera_cenarios(dados[ind3,:], S[i], N, p)
        @time π3, T3 , z3 = estima_contrato(D3)
        resultados3[:,i,r] = vcat(p, π3, T3, z3)
    end
end

#Export the results
cd("G:\\My Drive\\PUC-Rio\\PUC-Rio 2022-1\\Trabalho final ADR - Bruno Fanzeres\\Script\\Graphics")

z0_std_rep = Vector{Float64}()
z1_std_rep = Vector{Float64}()
z2_std_rep = Vector{Float64}()
z3_std_rep = Vector{Float64}()

for i in 1:length(S)
    z0_std_rep = push!(z0_std_rep, std(resultados0[6,i,:]))
end
for i in 1:length(S)
    z1_std_rep = push!(z1_std_rep, std(resultados1[6,i,:]))
end
for i in 1:length(S)
    z2_std_rep = push!(z2_std_rep, std(resultados2[6,i,:]))
end
for i in 1:length(S)
    z3_std_rep = push!(z3_std_rep, std(resultados3[6,i,:]))
end

z0_mean_rep = Vector{Float64}()
z1_mean_rep = Vector{Float64}()
z2_mean_rep = Vector{Float64}()
z3_mean_rep = Vector{Float64}()

for i in 1:length(S)
    z0_mean_rep = push!(z0_mean_rep, mean(resultados0[6,i,:]))
end
for i in 1:length(S)
    z1_mean_rep = push!(z1_mean_rep, mean(resultados1[6,i,:]))
end
for i in 1:length(S)
    z2_mean_rep = push!(z2_mean_rep, mean(resultados2[6,i,:]))
end
for i in 1:length(S)
    z3_mean_rep = push!(z3_mean_rep, mean(resultados3[6,i,:]))
end

using PyPlot, Printf, UnPack

macro name(arg)
   string(arg)
end

function plot_experiment(avg::Vector, std::Vector, scenarios::Vector)
    figure(figsize=(15, 7))
    ylabel("Mean VOF over replications", fontsize=15)
    xlabel("Number of scenarios", fontsize=15)
    PyPlot.plot(scenarios, avg, color="red", linewidth=2.0)
    PyPlot.fill_between(scenarios, avg.-std, avg.+std, alpha=0.5, linestyle="dashed", edgecolor="k", linewidth=2, antialiased=true)
    title("Convergence plot", fontsize=20)
end

plot_experiment(z0_mean_rep[1:end-1], z0_std_rep[1:end-1], S[1:end-1])
PyPlot.savefig(string(@name(z0_mean_rep)[1:2],@sprintf("_%04.d", S[end]),".png"))

plot_experiment(z1_mean_rep, z1_std_rep, S)
PyPlot.savefig(string(@name(z1_mean_rep)[1:2],@sprintf("_%04.d", S[end]),".png"))

plot_experiment(z2_mean_rep, z2_std_rep, S)
PyPlot.savefig(string(@name(z2_mean_rep)[1:2],@sprintf("_%04.d", S[end]),".png"))

plot_experiment(z3_mean_rep, z3_std_rep, S)
PyPlot.savefig(string(@name(z3_mean_rep)[1:2],@sprintf("_%04.d", S[end]),".png"))

#CSV.write(@sprintf("resultados_%02.d.csv", i), DataFrame(resultados[:,:,i], :auto))

#CSV.write("Cluster 0.csv", DataFrame(hcat(z0_mean_rep, z0_std_rep), [:mean, :std]))



############################################
## Analise do efeito da probabilidade "p" ##
############################################
#cd("G:\\My Drive\\PUC-Rio\\Trabalho final ADR - Bruno Fanzeres\\Script")

base = XLSX.readdata("Database.xlsx", "Sheet1!B2:N1339")
dados = Matrix{Float64}(base[:, [4, end]])

ind0 = findall(i -> i==0.0, dados[:, 2])
ind1 = findall(i -> i==1.0, dados[:, 2])
ind2 = findall(i -> i==2.0, dados[:, 2])
ind3 = findall(i -> i==3.0, dados[:, 2])

S = 1300
Rep = 10

N = 10
α = 0.95 #quantil do Cvar
λ = 0.5 #peso do cvar na função objetivo
τ = 0.3 #loading factor
Ti = 10
Tf = 150
M = [1e8, 1e8, 1e8]

p = collect(0.05:0.05:0.3)
#resultados = zeros(6, length(p), Rep)
resultados0 = zeros(6, length(p), Rep)
resultados1 = zeros(6, length(p), Rep)
resultados3 = zeros(6, length(p), Rep)
resultados2 = zeros(6, length(p), Rep)

for r in 1:Rep
    for i in 1:length(p)
        #D, cluster = gera_cenarios(dados, S, N, p[i])
        D0, cluster0 = gera_cenarios(dados[ind0,:], S, N, p[i])
        D1, cluster1 = gera_cenarios(dados[ind1,:], S, N, p[i])
        D2, cluster2 = gera_cenarios(dados[ind2,:], S, N, p[i])
        D3, cluster3 = gera_cenarios(dados[ind3,:], S, N, p[i])

        π, T , z = estima_contrato(D)
        π0, T0 , z0 = estima_contrato(D0)
        π1, T1 , z1 = estima_contrato(D1)
        π2, T2 , z2 = estima_contrato(D2)
        π3, T3 , z3 = estima_contrato(D3)

        resultados[:,i,r]  = vcat(p[i],π, T, z)
        resultados0[:,i,r] = vcat(p[i],π0, T0, z0)
        resultados1[:,i,r] = vcat(p[i],π1, T1, z1)
        resultados2[:,i,r] = vcat(p[i],π2, T2, z2)
        resultados3[:,i,r] = vcat(p[i],π3, T3, z3)
    end
end


cd("G:\\My Drive\\PUC-Rio\\Trabalho final ADR - Bruno Fanzeres\\Script\\Data Experiment")

for i in 1:Rep
    CSV.write(@sprintf("resultados_%02.d.csv", i), DataFrame(resultados[:,:,i], :auto))
end

for i in 1:Rep
    CSV.write(@sprintf("resultados0_%02.d.csv", i), DataFrame(resultados0[:,:,i], :auto))
end
for i in 1:Rep
    CSV.write(@sprintf("resultados1_%02.d.csv", i), DataFrame(resultados1[:,:,i], :auto))
end
for i in 1:Rep
    CSV.write(@sprintf("resultados2_%02.d.csv", i), DataFrame(resultados2[:,:,i], :auto))
end
for i in 1:Rep
    CSV.write(@sprintf("resultados3_%02.d.csv", i), DataFrame(resultados3[:,:,i], :auto))
end

D_teste, cluster_teste = gera_cenarios(dados, S, N, p)


CSV.write("contrato_unico.csv", DataFrame(resultados, :auto))
CSV.write("contrato0.csv", DataFrame(resultados0, :auto))
CSV.write("contrato1.csv", DataFrame(resultados1, :auto))
CSV.write("contrato2.csv", DataFrame(resultados2, :auto))
CSV.write("contrato3.csv", DataFrame(resultados3, :auto))
