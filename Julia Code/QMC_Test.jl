using Random, Distributions,  XLSX, DataFrames, Plots,  CSV, QuasiMonteCarlo, HypothesisTests

function compare_dist(y::Vector{Float64}, real_cluster::Vector{Float64}, sample::Dict, cluster::Union{Float64,String} , n::Vector{Int64})

    #criando a distribuicao acumulada de ylabel
    ordered_y = sort(y)
    Fy = zeros(length(ordered_y))
    for i in 1:length(y) 
        Fy[i] = i/length(y)
    end

    if cluster != "base"
        prop_real = sum(real_cluster .== cluster) / length(real_cluster)
    end

    mc_p = plot(ordered_y, Fy, label = "data", legend = :topleft, colour = "black", linewidth = 1.5)
           title!("Monte Carlo Sampling")
    qmc_p = plot(ordered_y, Fy,label = "data", legend = :topleft, colour = "black", linewidth = 1.5)
            title!("QMC LatinHypercubeSample")
    j = 1

    mc_pvalue = zeros(length(n))
    qmc_pvalue = zeros(length(n))

    mc_prop = zeros(length(n))
    qmc_prop = zeros(length(n))

    for i in n

        if cluster != "base"
            mc_y = sort(sample["MC"]["value"][i][findall(j -> j == cluster,sample["MC"]["cluster"][i])])
            qmc_y = sort(sample["QMC"]["value"][i][findall(j -> j == cluster,sample["QMC"]["cluster"][i])])

            mc_pvalue[j] = pvalue(ApproximateTwoSampleKSTest(y, sample["MC"]["value"][i][findall(j -> j == cluster,sample["MC"]["cluster"][i])]))
            qmc_pvalue[j] = pvalue(ApproximateTwoSampleKSTest(y, sample["QMC"]["value"][i][findall(j -> j == cluster,sample["QMC"]["cluster"][i])]))
        else
            mc_y = sort(sample["MC"]["value"][i])
            qmc_y = sort(sample["QMC"]["value"][i])

            mc_pvalue[j] = pvalue(ApproximateTwoSampleKSTest(y, sample["MC"]["value"][i]))
            qmc_pvalue[j] = pvalue(ApproximateTwoSampleKSTest(y, sample["QMC"]["value"][i]))
        end

        mc_fy = zeros(length(mc_y))
        qmc_fy = zeros(length(qmc_y))

        for j in 1:length(mc_y)
            mc_fy[j] = j/length(mc_y)
        end

        for j in 1:length(qmc_y)
            qmc_fy[j] = j/length(qmc_y)
        end

        mc_p = plot!(mc_p, mc_y, mc_fy, label = "n = $i", legend = false)
        qmc_p = plot!(qmc_p, qmc_y, qmc_fy,label = "n = $i", legend = false)
       
        if cluster != 4.0
            mc_prop[j] = sum(sample["MC"]["cluster"][i] .== cluster) / i
            qmc_prop[j] = sum(sample["QMC"]["cluster"][i] .== cluster) / i
        end

        j += 1
    end

    pvalue_p = plot(n, mc_pvalue, label = "MC")
               plot!(n, qmc_pvalue, label = "QMC", legend = :bottomright)
               title!("KS Test: P-value")

    if cluster != 4.0
        prop_p = plot(n, mc_prop, label = "MC")
                plot!(n, qmc_prop, label = "QMC", legend = :bottomright)
                hline!(n, [prop_real], label = "Real Prop.", colour = "black")
                title!("Proportion convergence")
        
        return mc_p, qmc_p, pvalue_p, prop_p
    else
        return mc_p, qmc_p, pvalue_p, nothing
    end
end

base = XLSX.readdata("Databases/Database.xlsx", "Sheet1!B2:N1339")
dados = Matrix{Float64}(base[:, [4, end]])

ordered_y = sort(dados[:, 1])

Fy = zeros(length(ordered_y))
for i in 1:1338 
    Fy[i] = i/1338
end

sample = Dict()
sample["MC"] = Dict()
sample["QMC"] = Dict()
sample["MC"]["value"] = Dict()
sample["MC"]["cluster"] = Dict()
sample["QMC"]["value"] = Dict()
sample["QMC"]["cluster"] = Dict()


n = collect(100:100:2000)

for j in 1:length(n)

    println("j = $j")
    MC_sample = zeros(n[j])
    MC_cluster = zeros(n[j])

    QMC_sample_latin = zeros(n[j])
    QMC_cluster_latin = zeros(n[j])

    MC_idx = rand(Uniform(minimum(Fy),1), n[j])
    QMC_idx_latin = QuasiMonteCarlo.sample(n[j],minimum(Fy),1,LatinHypercubeSample())
    
    for i in 1:n[j]
        mc_idx = maximum(findall(k -> k <= MC_idx[i], Fy))
        qmc_idx_latin = maximum(findall(k -> k <= QMC_idx_latin[i], Fy))

        MC_sample[i] = ordered_y[mc_idx]
        MC_cluster[i] = dados[findall(m -> m == MC_sample[i], dados[:, 1])[1],2]
    
        QMC_sample_latin[i] = ordered_y[qmc_idx_latin]
        QMC_cluster_latin[i] = dados[findall(m -> m == QMC_sample_latin[i], dados[:, 1])[1],2]
    end
    sample["MC"]["value"][n[j]] = MC_sample
    sample["MC"]["cluster"][n[j]] = MC_cluster
    sample["QMC"]["value"][n[j]] = QMC_sample_latin
    sample["QMC"]["cluster"][n[j]] = QMC_cluster_latin
end

#cluster 0

cluster0 = dados[findall(i -> i == 0.0, dados[:, 2]), :]
cluster1 = dados[findall(i -> i == 1.0, dados[:, 2]), :]
cluster2 = dados[findall(i -> i == 2.0, dados[:, 2]), :]
cluster3 = dados[findall(i -> i == 3.0, dados[:, 2]), :]

mc_c0, qmc_c0, c0_pvalue, c0_prop = compare_dist(cluster0[:, 1],dados[:, 2], sample, 0.0, n)
mc_c1, qmc_c1, c1_pvalue, c1_prop= compare_dist(cluster1[:, 1],dados[:, 2], sample, 1.0, n)
mc_c2, qmc_c2, c2_pvalue, c2_prop = compare_dist(cluster2[:, 1],dados[:, 2], sample, 2.0, n)
mc_c3, qmc_c3, c3_pvalue, c3_prop = compare_dist(cluster3[:, 1], dados[:, 2],sample, 3.0, n)
mc_base, qmc_base, pvalue_base, prop_base = compare_dist(dados[:, 1],dados[:, 2], sample, 4.0, n)





p1 = plot(cluster0_ordered_y, cluster0_Fy)

for i in 1:length(n)
plot!(p1, sample["MC"][""])

end



p1 = plot(n,mc_mean_std, colour = "black", label = "Monte Carlo")
plot!(n,qmc_mean_std_unif, label = "QMC - Uniforme")
#plot!(qmc_mean_std_sobol)
plot!(n,qmc_mean_std_latin,  label = "QMC - LatinHypercubeSample")
plot!(n, qmc_mean_std_lattice, label = "QMC - LatticeRuleSample")
ylabel!("Variância")
xlabel!("Tamanho da amostra")

var(rand(Uniform(minimum(Fy),1), n[j]))
var(QMC_sampling)

savefig(p1, "teste_QMC.png")