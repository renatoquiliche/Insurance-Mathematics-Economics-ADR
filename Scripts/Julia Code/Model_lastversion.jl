using JuMP,  CSV, DataFrames, Distributions, StatsBase, Random, Plots, QuasiMonteCarlo, HiGHS, StatsPlots, HypothesisTests



function estima_contrato(D::Matrix, α, λ, τ, Ti, Tf, parallel)
    S,N = size(D)
    p = ones(S) * 1/S

    model = Model()
    model = Model(optimizer_with_attributes(HiGHS.Optimizer,  "parallel" => parallel))
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
    @constraint(model, π ≤ τ*sum(p[s]*I[s] for s in 1:S)) #restringindo o premio em funcao da indenizacao media
    #@constraint(model, π ≤ 2.5*sum(p[s]*I[s] for s in 1:S)) #restringindo o premio em funcao da indenizacao media
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
    s_aux = zeros(sample_size, N)
    cluster = Int64.(zeros(sample_size))
    ran = zeros(sample_size,N)
    idx = zeros(sample_size, N)
    for n in 1:N
        ran[:, n] = rand(sample_size)
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
            #ind_charge = rand(Bernoulli(p), 1)
            #ind_charge = 1
            #idx = randperm(1)
            P = ((idx[j,i]-ran[j,i])/sample_size)*100 # probability of the cdf
            s_aux[j,i] = (percentile(data[ind_cluster,1],P) / 1000)[1]; # Trick for sampling in thousands
            #hist_s = histogram(y, label="Sampled", bins=:sqrt)
            #hist_s = histogram!(s, label="Re-sampled QMC", fillalpha=0.7, bins=:sqrt)
        end
    end
    P = ((idx-ran)/sample_size)
    QMC_Bern = quantile(Bernoulli(p), P) #Transformada inversa
    s = QMC_Bern.*s_aux
    return s, cluster
end
"""
Test convergencia 2000*

N=3
p=0.3
y_qmc = Vector{Float64}()
grid = 10:10:2000
for s in grid
    y_qmc = push!(y_qmc,mean(QMC_LHS_empirical(Data_smoker, s, N, p, 1)[1]))
end
plot(collect(grid), y_qmc, label="QMC-LHS")
"""

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

HyperP = Dict()
#Base scenario
HyperP["N"] = 3
HyperP["α"] = 0.95
HyperP["λ"] = 0.25
HyperP["τ"] = 2.5
HyperP["Ti"] = 1.0
HyperP["Tf"] = 1.5
HyperP["M"] = [1e8, 1e8, 1e8]
HyperP["p"] = 0.1
HyperP["sample_size"] = 700

M = [1e8, 1e8, 1e8]

function optimal_contract_point(N, α, λ, τ, Ti, Tf, p, sample_size, multithreading)

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


    D_unique, cluster_unique = QMC_LHS_empirical(Data, sample_size, N, p, 2)
    D_smoker, cluster_smoker = QMC_LHS_empirical(Data_smoker, sample_size, N, p, 1)
    D_nsmoker, cluster_nsmoker = QMC_LHS_empirical(Data_nsmoker, sample_size, N, p, 1)

    D1, cluster1 = QMC_LHS_empirical(Data1, sample_size,N, p,2) #Non-smoker
    D2, cluster2 = QMC_LHS_empirical(Data2, sample_size,N, p,2) #Soker+Obese
    D3, cluster3 = QMC_LHS_empirical(Data3, sample_size,N, p,2) #Smoker+Non Obese

    D = [D_unique, D_smoker, D_nsmoker, D1,D2,D3]

    for j in 1:6
        time = @elapsed π0, T0 , z0, exp, cvar = estima_contrato(D[j], α, λ, τ, Ti, Tf, multithreading)
        resultados[:,j] = vcat(time,p, π0, T0, z0, exp, cvar)
    end

    resultados

    dict_contracts = Dict()
    dict_contracts["1"] = Dict()
    dict_contracts["2"] = Dict()
    dict_contracts["unique"] = resultados[3:6,1]
    dict_contracts["1"][1] =  resultados[3:6,2] # Smoker
    dict_contracts["1"][0] =  resultados[3:6,3] # Non-smoker
    dict_contracts["2"][1] =  resultados[3:6,4] # Non-smoker
    dict_contracts["2"][2] =  resultados[3:6,5] # Smoker + Non Obese
    dict_contracts["2"][3] =  resultados[3:6,6] # Smoker + Obese

    r = resultados'


    return dict_contracts, r

end



"""EXPERIMENTS"""


#optimal_contract_point(3, 0.95, 0.25, 2.1, 0.1, 2.0, 0.1, 1200)

for p in 0.1:0.05:2.0
    print(p)
    CSV.write("ExperimentBase_Ti_$Ti.csv", optimal_contract_point(3, 0.95, 0.25, 2.5, Ti, 2.0, p, 700))
end

for Ti in 0.1:0.05:2.0
    print(Ti)
    CSV.write("ExperimentBase_Ti_$Ti.csv", optimal_contract_point(3, 0.95, 0.25, 2.5, Ti, 2.0, 0.1, 700))
end

τ
# Loading factor

@time Threads.@threads for τ in 1.6:0.05:3.5
    println("Experiment: $τ")
    CSV.write("ExperimentBase_lf_$τ.csv", optimal_contract_point(3, 0.90, 0.10, τ, 0.1, 2.0, 0.1, 2000))
end

N=3
p = 0.2
dict_contracts = optimal_contract_point(3, 0.85, 0.50, 2.0, 0.1, 2.0, p, 2000, "on")[1]

test_size = 3000

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

D_unique_test, cluster_unique = QMC_LHS_empirical(Data, test_size, N, p, 2)
D_smoker_test, cluster_smoker = QMC_LHS_empirical(Data_smoker, test_size, N, p, 1)
D_nsmoker_test, cluster_nsmoker = QMC_LHS_empirical(Data_nsmoker, test_size, N, p, 1)

D1_test, cluster1 = QMC_LHS_empirical(Data1, test_size,N, p,2) #Non-smoker
D2_test, cluster2 = QMC_LHS_empirical(Data2, test_size,N, p,2) #Soker+Obese
D3_test, cluster3 = QMC_LHS_empirical(Data3, test_size,N, p,2) #Smoker+Non Obese

#D_teste, cluster_teste = QMC_LHS_empirical(Data, test_size,N, p, 2)
#cluster_teste1 = zeros(test_size)
S,N = size(D_unique_test)

receita = zeros(S)
indenizacao_aux = zeros(S, N) #Matriz de indemnizacao
indenizacao = zeros(S)
for s in 1:S
    for n in 1:N
        indenizacao_aux[s, n] = min(D_unique_test[s, n], dict_contracts["unique"][3]) #T2
        indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict_contracts["unique"][4]) #T3
        receita[s] = dict_contracts["unique"][1] .- indenizacao[s]
    end
end
for s in 1:S
    for n in 1:N
        indenizacao_aux[s, n] = min(D_smoker_test[s, n], dict_contracts["1"][1][3]) #T2
        indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict_contracts["1"][1][4]) #T3
        receita[s] = dict_contracts["1"][1][1] .- indenizacao[s]
    end
end
for s in 1:S
    for n in 1:N
        indenizacao_aux[s, n] = min(D_nsmoker_test[s, n], dict_contracts["1"][0][3]) #T2
        indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict_contracts["1"][0][4]) #T3
        receita[s] = dict_contracts["1"][0][1] .- indenizacao[s]
    end
end
# Por Cluster
# C1
for s in 1:S
    for n in 1:N
        indenizacao_aux[s, n] = min(D1_test[s, n], dict_contracts["2"][1][3]) #T2
        indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict_contracts["2"][1][4]) #T3
        receita[s] = dict_contracts["2"][1][1] .- indenizacao[s]
    end
end
# C2
for s in 1:S
    for n in 1:N
        indenizacao_aux[s, n] = min(D2_test[s, n], dict_contracts["2"][2][3]) #T2
        indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict_contracts["2"][2][4]) #T3
        receita[s] = dict_contracts["2"][2][1] .- indenizacao[s]
    end
end
# C3
for s in 1:S
    for n in 1:N
        indenizacao_aux[s, n] = min(D3_test[s, n], dict_contracts["2"][3][3]) #T2
        indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict_contracts["2"][3][4]) #T3
        receita[s] = dict_contracts["2"][3][1] .- indenizacao[s]
    end
end

summarystats(receita)
density(receita)

for s in 1:S
    for n in 1:N
        if unique
            indenizacao_aux[s, n] = min(D_teste[s, n], dict["unique"][3]) #T2
        else
            indenizacao_aux[s, n] = min(D_teste[s, n], dict_contracts[type][cluster[s]][3]) #T2
        end 
    end
    if unique
        indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict["unique"][4]) #T3
        receita[s] = dict["unique"][1] .- indenizacao[s]
    else
        indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict[type][cluster[s]][4]) #T3
        receita[s] = dict[type][cluster[s]][1] .- indenizacao[s]
    end
end

function calcula_receita(D::Matrix, cluster::Vector, dict::Dict, unique::Bool, type::String)

    S,N = size(D)

    receita = zeros(S)
    indenizacao_aux = zeros(S, N) #Matriz de indemnizacao
    indenizacao = zeros(S) #Indemnizacao total por escenario

    dict_contracts["unique"][cluster_teste[100]][3]
    for s in 1:S
        for n in 1:N
            if unique
                indenizacao_aux[s, n] = min(D_teste[s, n], dict["unique"][3]) #T2
            else
                indenizacao_aux[s, n] = min(D_teste[s, n], dict_contracts[type][cluster[s]][3]) #T2
            end 
        end
        if unique
            indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict["unique"][4]) #T3
            receita[s] = dict["unique"][1] .- indenizacao[s]
        else
            indenizacao[s] = min(sum(indenizacao_aux[s,:]), dict[type][cluster[s]][4]) #T3
            receita[s] = dict[type][cluster[s]][1] .- indenizacao[s]
        end
    end

    return receita, indenizacao
end

cluster_teste1 = zeros(test_size)
for i in 1:test_size
    if cluster_teste[i] == 2 || cluster_teste[i] == 3
        cluster_teste1[i] = 1
    else
        cluster_teste1[i] = 0
    end
end

receita_unique, indenizacao_unique = calcula_receita(D_teste, cluster_teste, dict_contracts, true, "unique")


receita_smoker, indenizacao_smoker = calcula_receita(D_teste, cluster_teste1, dict_contracts, false, "1")
receita_cluster, indenizacao_cluster = calcula_receita(D_teste, cluster_teste, dict_contracts, false, "2")

density(receita_unique, label = "Contrato único", linewidth = 1.5)
density!(receita_smoker, label = "Contrato por hábito de fumar", legend= :topright , linewidth = 1.5)
density!(receita_cluster, label = "Contrato por cluster", legend= :topright , linewidth = 1.5)
title!("Comparativo das receitas")

# Graph
receita = vcat(receita_unique, receita_smoker, receita_cluster)
contrato = vcat(repeat(["Contrato único"], inner = test_size), repeat(["Dois contratos"], inner = test_size), repeat(["Tres contratos"], inner = test_size))


StatsPlots.violin(contrato, log.(receita.+1000), color = "gray", alpha = 1)
boxplot!(contrato, log.(receita.+1000), colour = "blue", legend = false)
hline!(hline!([log(1000)]))
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