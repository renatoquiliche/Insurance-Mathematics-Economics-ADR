using JuMP,  CSV, DataFrames, Distributions, StatsBase, Random, Plots, COPT, QuasiMonteCarlo, HiGHS, StatsPlots, HypothesisTests



function estima_contrato(D::Matrix)
    S,N = size(D)
    p = ones(S) * 1/S

    model = Model(HiGHS.Optimizer)
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

    @objective(model, Max, π - (((1 - λ)*sum(p[s] * I[s] for s in 1:S)) + λ*(v + sum(p[s]*β[s]/(1 - α) for s in 1:S))))

    #@constraint(model, π ≤ (1 + τ)*sum(p[s]*I[s] for s in 1:S)) #A.2 (premio maior do que a idenizacao media)
    @constraint(model, π ≤ 2.5*sum(p[s]*I[s] for s in 1:S)) #restringindo o premio em funcao da indenizacao media
    @constraint(model, π ≥ 0.7*sum(p[s]*I[s] for s in 1:S)) #restringindo o premio em funcao da indenizacao media
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
    @constraint(model, [s in 1:S], β[s] ≥ (I[s] - v)) #A.23

    optimize!(model)

    expected = sum(p[s] * value.(I)[s] for s in 1:S)
    cvar = value(v) + sum(p[s]*value.(β)[s]/(1 - α) for s in 1:S)

    return value(π), value.(T), objective_value(model), expected, cvar
end

function MC_bootstrap(y, sample_size)
    x = ones(sample_size,1)

    StatsBase.sample!(y, x, replace=true, ordered=false)
    hist_x = histogram(y, label="Sampled", bins=:sqrt)
    hist_x = histogram!(x, label="Re-sampled MC", fillalpha=0.7, bins=:sqrt)
    return x, hist_x
end

function QMC_LHS_empirical(data, sample_size, N, p, type)
    s = zeros(sample_size, N)
    cluster = Int64.(zeros(sample_size))
    ran = zeros(sample_size,N)
    idx = zeros(sample_size, N)
    for n in 1:N
        ran[:, n] = QuasiMonteCarlo.sample(sample_size,0,1,LatinHypercubeSample())
        idx[:, n] = randperm(sample_size)
    end

    for j in 1:sample_size
        if type == 1
            cluster[j] = rand(data.Cluster, 1)[1]
            ind_cluster = findall(k -> k==cluster[j], data.Cluster)
        else
            cluster[j] = rand(data.Cluster, 1)[1]
            ind_cluster = findall(k -> k==cluster[j], data.Cluster)
        end

        for i in 1:N
            #ran = rand(Uniform(0, 1), sample_size);
            #ran = QuasiMonteCarlo.sample(sample_size,0,1,LatinHypercubeSample())
            #s = zeros(sample_size,1)
            ind_charge = rand(Bernoulli(p), 1)
            #idx = randperm(1)
            P = ((idx[j,i]-ran[j,i])/sample_size)*100 # probability of the cdf
            s[j,i] = (ind_charge * percentile(data[ind_cluster,1],P) / 1000)[1]; # Trick for sampling
            #hist_s = histogram(y, label="Sampled", bins=:sqrt)
            #hist_s = histogram!(s, label="Re-sampled QMC", fillalpha=0.7, bins=:sqrt)
        end
    end
    return s, cluster
end

function calcula_receita(D::Matrix, cluster::Vector, dict::Dict, unique::Bool, type::String)

    S,N = size(D)

    receita = zeros(S)
    indenizacao_aux = zeros(S, N)
    indenizacao = zeros(S)

    for s in 1:S
        for n in 1:N
            if unique
                indenizacao_aux[s, n] = min(D_teste[s, n], dict["unique"][3])
            else
                indenizacao_aux[s, n] = min(D_teste[s, n], dict[type][cluster[s]][3])
            end
        end
        if unique
            indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict["unique"][4])
            receita[s] = dict["unique"][1] .- indenizacao[s]
        else
            indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict[type][cluster[s]][4])
            receita[s] = dict[type][cluster[s]][1] .- indenizacao[s]
        end
    end

    return receita, indenizacao
end

Data = CSV.read("Databases/contracts.csv", DataFrame)

unique(Data[!,["Category", "Cluster"]])

# Hyperparameters
N = 2 #No de accidentes
α = 0.95 #quantil do Cvar
λ = 0.25 #peso do cvar na função objetivo
τ = 1.0 #loading factor
Ti = 0.1
Tf = 0.5
M = [1e8, 1e8, 1e8]
p = 0.05

#Sorting out data for each cluster

# Getting index
ind_smoker = findall(i -> i > 1, Data.Cluster)
ind_nsmoker = findall(i -> i == 1, Data.Cluster)
#ind0 = findall(i -> i == 0, Data[:,8])
ind1 = findall(i -> i == 1, Data.Cluster)
ind2 = findall(i -> i == 2, Data.Cluster)
ind3 = findall(i -> i == 3, Data.Cluster)

Data_smoker = Data[ind_smoker,:]
Data_nsmoker = Data[ind_nsmoker,:]
#Data0 = Data[ind0,:]
Data1 = Data[ind1, :]
Data2 = Data[ind2, :]
Data3 = Data[ind3, :]

resultados = zeros(9, 6)

Random.seed!(123)
sample_size = 500

D_unique, cluster_unique = QMC_LHS_empirical(Data, sample_size, N, p, 2)
D_smoker, cluster_smoker = QMC_LHS_empirical(Data_smoker, sample_size, N, p, 1)
D_nsmoker, cluster_nsmoker = QMC_LHS_empirical(Data_nsmoker, sample_size, N, p, 1)

D1, cluster1 = QMC_LHS_empirical(Data1, sample_size,N, p,2) #Non-smoker
D2, cluster2 = QMC_LHS_empirical(Data2, sample_size,N, p,2) #Soker+Obese
D3, cluster3 = QMC_LHS_empirical(Data3, sample_size,N, p,2) #Smoker+Non Obese

D = [D_unique, D_smoker, D_nsmoker, D1,D2,D3]

for j in 1:6
    time = @elapsed π0, T0 , z0, exp, cvar = estima_contrato(D[j])
    resultados[:,j] = vcat(time,p, π0, T0, z0, exp, cvar)
end



dict_contracts = Dict()
dict_contracts["1"] = Dict()
dict_contracts["2"] = Dict()
dict_contracts["unique"] = resultados[3:6,1]
dict_contracts["1"][1] =  resultados[3:6,2] # Smoker
dict_contracts["1"][0] =  resultados[3:6,3] # Non-smoker
dict_contracts["2"][0] =  resultados[3:6,4] # Non-smoker
dict_contracts["2"][1] =  resultados[3:6,5] # Smoker + Non Obese
dict_contracts["2"][2] =  resultados[3:6,6] # Smoker + Obese

r = resultados'

#D_teste[1,1]

Random.seed!(321)
D_teste, cluster_teste = QMC_LHS_empirical(Data, 300,N, p, 2)
cluster_teste1 = zeros(300)

for i in 1:300
    if cluster_teste[i] == 1 || cluster_teste[i] == 3
        cluster_teste1[i] = 1
    else
        cluster_teste1[i] = 0
    end
end

receita_unique, indenizacao_unique = calcula_receita(D_teste, cluster_teste, dict_contracts, true, "2")
receita_smoker, indenizacao_smoker = calcula_receita(D_teste, cluster_teste1, dict_contracts, false, "1")
receita_cluster, indenizacao_cluster = calcula_receita(D_teste, cluster_teste, dict_contracts, false, "2")

density(receita_unique, label = "Contrato único", linewidth = 1.5)
density!(receita_smoker, label = "Contrato por hábito de fumar", legend= :topleft , linewidth = 1.5)
density!(receita_cluster, label = "Contrato por cluster", legend= :topleft , linewidth = 1.5)
title!("Comparativo das receitas")

receita = vcat(receita_unique, receita_smoker, receita_cluster)
contrato = vcat(repeat(["Contrato único"], inner = 300), repeat(["Dois contratos"], inner = 300), repeat(["Quatro contratos"], inner = 300))

StatsPlots.violin(contrato, receita, color = "gray", alpha = 0.5)
boxplot!(contrato, receita, colour = "blue", legend = false)
title!("Distribuição das receitas (p = $p)")
yaxis!("Receita")

summarystats(receita_unique)
summarystats(receita_smoker)
summarystats(receita_cluster)

#teste de hipoteses H0: medianas iguais

ApproximateSignedRankTest(receita_unique, receita_smoker)
ApproximateSignedRankTest(receita_unique, receita_cluster)
ApproximateSignedRankTest(receita_smoker, receita_cluster)


ExactSignedRankTest(receita_unique, receita_smoker)
ExactSignedRankTest(receita_unique, receita_cluster)
ExactSignedRankTest(receita_smoker, receita_cluster)







boxplot(receita_unique, label = "Contrato único", linewidth = 1.5)
boxplot!(receita_smoker, label = "Contrato por hábito de fumar", legend= :topleft , linewidth = 1.5)
boxplot!(receita_cluster, label = "Contrato por cluster", legend= :topleft , linewidth = 1.5)
StatsPlots.violin!(receita_unique, label = "Contrato único", linewidth = 1.5)
StatsPlots.violin!(receita_smoker, label = "Contrato por hábito de fumar", legend= :topleft , linewidth = 1.5)
StatsPlots.violin!(receita_cluster, label = "Contrato por cluster", legend= :topleft , linewidth = 1.5)
title!("Comparativo das receitas")


StatsPlots.violin(receita_unique, label = "Contrato único", linewidth = 1.5)
boxplot!(receita_unique, label = "Contrato único", linewidth = 1.5)

summarystats(receita_unique)
summarystats(receita_smoker)
summarystats(receita_cluster)


Data[ind_smoker, :]






Plots.plot(S1, resultados[6,1,:])
Plots.hline!([12.508336242883423])
Plots.savefig("ConvergenceRep1.png")
CSV.write("ConvergenceRep1.csv",  Tables.table(resultados[6,1,:]), writeheader=true)

S2 = collect(5:5:300)
Rep2 = 100

resultados = zeros(6, Rep2, length(S2))

# Hyperparameters
N = 10
α = 0.95 #quantil do Cvar
λ = 0.5 #peso do cvar na função objetivo
τ = 0.3 #loading factor
Ti = 10
Tf = 150
M = [1e8, 1e8, 1e8]
p = 0.3

for j in 1:length(S2)
    for r in 1:Rep2
        D = QMC_LHS_empirical(charges, S2[j])[1]
        @time π0, T0 , z0 = estima_contrato(D)
        resultados[:,r,j] = vcat(p, π0, T0, z0)
    end
end


Plots.plot(S2, resultados[6,1,:])
Plots.hline!([12.508336242883423])
Plots.savefig("ConvergenceRep100.png")

for j in 1:length(S2)
    CSV.write("ConvergenceRep100_$j.csv",  Tables.table(resultados[6,:,j]), writeheader=true)
end
