using Random, Distributions,  XLSX, DataFrames, Plots,  CSV, QuasiMonteCarlo, HypothesisTests, Statistics

#Ths function receives the charges of a cluster, a vector of the real cluster of the data, the dict of samples, the cluster that you want to analysis,
# the version of the cluster classification (v0 - algorithm based, v1 - smoker, v2 - smoker, BMI, age) and the size os the samples.

#The ideia is to return plots comparing the samples distributions and teh real distribution.
function compare_dist(y::Vector{Float64}, real_cluster::Vector{Float64}, sample::Dict, cluster::Union{Float64,String}, version::String, n::Vector{Int64})

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
            mc_y = sort(sample["MC"]["value"][i][findall(j -> j == cluster,sample["MC"]["cluster"][version][i])])
            qmc_y = sort(sample["QMC"]["value"][i][findall(j -> j == cluster,sample["QMC"]["cluster"][version][i])])

            mc_pvalue[j] = pvalue(ApproximateTwoSampleKSTest(y, sample["MC"]["value"][i][findall(j -> j == cluster,sample["MC"]["cluster"][version][i])]))
            qmc_pvalue[j] = pvalue(ApproximateTwoSampleKSTest(y, sample["QMC"]["value"][i][findall(j -> j == cluster,sample["QMC"]["cluster"][version][i])]))
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

        if cluster != "base"
            mc_prop[j] = sum(sample["MC"]["cluster"][version][i] .== cluster) / i
            qmc_prop[j] = sum(sample["QMC"]["cluster"][version][i] .== cluster) / i
        end

        j += 1
    end

    pvalue_p = plot(n, mc_pvalue, label = "MC")
               plot!(n, qmc_pvalue, label = "QMC", legend = :bottomright)
               title!("KS Test: P-value")

    if cluster != "base"
        prop_p = plot(n, mc_prop, label = "MC")
                plot!(n, qmc_prop, label = "QMC", legend = :bottomright)
                hline!(n, [prop_real], label = "Real Prop.", colour = "black")
                title!("Proportion convergence")

        return mc_p, qmc_p, pvalue_p, prop_p
    else
        return mc_p, qmc_p, pvalue_p, nothing
    end
end

base_v1 = XLSX.readdata("Databases/Database.xlsx", "Sheet1!B2:O1339")
base_v2 = CSV.read("Databases/contracts.csv", DataFrame)
dados = Matrix{Float64}(base_v1[:, [4, end-1]])

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
sample["MC"]["cluster"]["v0"] = Dict()
sample["MC"]["cluster"]["v1"] = Dict()
sample["MC"]["cluster"]["v2"] = Dict()
sample["QMC"]["value"] = Dict()
sample["QMC"]["cluster"] = Dict()
sample["QMC"]["cluster"]["v0"] = Dict()
sample["QMC"]["cluster"]["v1"] = Dict()
sample["QMC"]["cluster"]["v2"] = Dict()

#Random.seed!(123)
#seed_mc = rand(collect(1:1000), 20) #seeds used in the Monte carlo sampling
#Random.seed!(321)
#seed_qmc = rand(collect(1:1000), 20) #seeds used in the Quasi-Monte carlo sampling

MC_sd1 = zeros(10,20)
QMC_sd1 = zeros(10,20)

MC_sd2 = zeros(10,20)
QMC_sd2 = zeros(10,20)

n = collect(100:100:2000)

for s in 1:10
    for j in 1:length(n)

        println("j = $j")
        MC_sample = zeros(n[j])
        MC_cluster_v0 = zeros(n[j]) #cluster by algorithm
        MC_cluster_v1 = zeros(n[j]) #cluster by smoker variable
        MC_cluster_v2 = zeros(n[j]) #cluster by smoder, BMI and age variables

        QMC_sample_latin = zeros(n[j])
        QMC_cluster_v0_latin = zeros(n[j])
        QMC_cluster_v1_latin = zeros(n[j])
        QMC_cluster_v2_latin = zeros(n[j])

        #Random.seed!(seed_mc[j])
        MC_idx = rand(Uniform(0,1), n[j])
        MC_sd1[s,j] = std(MC_idx)
        #Random.seed!(seed_qmc[j])
        QMC_idx_latin = QuasiMonteCarlo.sample(n[j],0,1,LatinHypercubeSample())
        QMC_sd1[s,j] = std(QMC_idx_latin)

        for i in 1:n[j]

            if MC_idx[i] < Fy[1]
                mc_idx = 1
            else
                mc_idx = maximum(findall(k -> k <= MC_idx[i], Fy)) #obtaining the index of sampled value in the real cumulative distribution
            end

            if QMC_idx_latin[i] < Fy[1]
                qmc_idx_latin = 1
            else
                qmc_idx_latin = maximum(findall(k -> k <= QMC_idx_latin[i], Fy)) #obtaining the index of sampled value in the real cumulative distribution
            end

            MC_sample[i] = ordered_y[mc_idx] #obtaining the value sampled by Monte carlo
            MC_cluster_v0[i] = dados[findall(m -> m == MC_sample[i], dados[:, 1])[1],2] #obtaining the cluster of the sampled value by Monte carlo
            MC_cluster_v1[i] = base_v2[findall(m -> m == MC_sample[i], base_v2[:, 1])[1],7] #obtaining the cluster of the sampled value by Monte carlo
            MC_cluster_v2[i] = base_v2[findall(m -> m == MC_sample[i], base_v2[:, 1])[1],6] #obtaining the cluster of the sampled value by Monte carlo

            QMC_sample_latin[i] = ordered_y[qmc_idx_latin]
            QMC_cluster_v0_latin[i] = dados[findall(m -> m == QMC_sample_latin[i], dados[:, 1])[1],2] #obtaining the cluster of the sampled value by Quasi-Monte carlo
            QMC_cluster_v1_latin[i] = base_v2[findall(m -> m == QMC_sample_latin[i], base_v2[:, 1])[1],7] #obtaining the cluster of the sampled value by Quasi-Monte carlo
            QMC_cluster_v2_latin[i] = base_v2[findall(m -> m == QMC_sample_latin[i], base_v2[:, 1])[1],6] #obtaining the cluster of the sampled value by Quasi-Monte carlo
        end
        sample["MC"]["value"][n[j]] = MC_sample
        sample["MC"]["cluster"]["v0"][n[j]] = MC_cluster_v0
        sample["MC"]["cluster"]["v1"][n[j]] = MC_cluster_v1
        sample["MC"]["cluster"]["v2"][n[j]] = MC_cluster_v2
        sample["QMC"]["value"][n[j]] = QMC_sample_latin
        sample["QMC"]["cluster"]["v0"][n[j]] = QMC_cluster_v0_latin
        sample["QMC"]["cluster"]["v1"][n[j]] = QMC_cluster_v1_latin
        sample["QMC"]["cluster"]["v2"][n[j]] = QMC_cluster_v2_latin

        MC_sd2[s,j] = std(MC_sample)
        QMC_sd2[s,j] = std(QMC_sample_latin)
    end
end

#Montando gráficos...
p1 = scatter(collect(100:100:2000), MC_sd1[1,:], color = "red", legend = false)
title!("Monte Carlo - sample scale")
xaxis!("Sample size")
yaxis!("Standard deviation")

p2 = scatter(collect(100:100:2000), QMC_sd1[1,:], color = "blue", legend = false)
title!(" Quasi-Monte Carlo - sample scale")
xaxis!("Sample size")
yaxis!("Standard deviation")

p3 = scatter(collect(100:100:2000), MC_sd2[1,:], color = "red", legend = false)
title!("Monte Carlo - Data scale")
xaxis!("Sample size")
yaxis!("Standard deviation")
p4 = scatter(collect(100:100:2000), QMC_sd2[1,:], color = "blue", legend = false)
title!("Quasi-Monte Carlo - Data scale")
xaxis!("Sample size")
yaxis!("Standard deviation")

for i in 2:10
    scatter!(p1, collect(100:100:2000), MC_sd1[i,:],color = "red", legend = false)
    scatter!(p2,collect(100:100:2000), QMC_sd1[i,:], color = "blue", legend = false)
    scatter!(p3, collect(100:100:2000), MC_sd2[i,:],color = "red", legend = false)
    scatter!(p4,collect(100:100:2000), QMC_sd2[i,:], color = "blue", legend = false)
end

#complete database
mc_base, qmc_base, pvalue_base, prop_base = compare_dist(dados[:, 1],dados[:, 2], sample, "base", "v0", n)

# V0 clustering
cluster0 = dados[findall(i -> i == 0.0, dados[:, 2]), :]
cluster1 = dados[findall(i -> i == 1.0, dados[:, 2]), :]
cluster2 = dados[findall(i -> i == 2.0, dados[:, 2]), :]
cluster3 = dados[findall(i -> i == 3.0, dados[:, 2]), :]

mc_c0, qmc_c0, c0_pvalue, c0_prop = compare_dist(cluster0[:, 1],dados[:, 2], sample, 0.0, "v0", n)
mc_c1, qmc_c1, c1_pvalue, c1_prop= compare_dist(cluster1[:, 1],dados[:, 2], sample, 1.0,"v0", n)
mc_c2, qmc_c2, c2_pvalue, c2_prop = compare_dist(cluster2[:, 1],dados[:, 2], sample, 2.0, "v0", n)
mc_c3, qmc_c3, c3_pvalue, c3_prop = compare_dist(cluster3[:, 1], dados[:, 2],sample, 3.0, "v0", n)

#v1 clustering
cluster0 = Matrix{Float64}(base_v2[findall(i -> i == 0.0, base_v2[:, 7]), [1,7]])
cluster1 = Matrix{Float64}(base_v2[findall(i -> i == 1.0, base_v2[:, 7]), [1,7]])

mc_c0, qmc_c0, c0_pvalue, c0_prop = compare_dist(cluster0[:, 1],Vector{Float64}(base_v2[:, 7]), sample, 0.0, "v1", n)
mc_c1, qmc_c1, c1_pvalue, c1_prop= compare_dist(cluster1[:, 1],Vector{Float64}(base_v2[:, 7]), sample, 1.0,"v1", n)

#v2 clustering
cluster0 = Matrix{Float64}(base_v2[findall(i -> i == 0.0, base_v2[:, 6]), [1,6]])
cluster1 = Matrix{Float64}(base_v2[findall(i -> i == 1.0, base_v2[:, 6]), [1,6]])
cluster2 = Matrix{Float64}(base_v2[findall(i -> i == 2.0, base_v2[:, 6]), [1,6]])
cluster3 = Matrix{Float64}(base_v2[findall(i -> i == 3.0, base_v2[:, 6]), [1,6]])
cluster4 = Matrix{Float64}(base_v2[findall(i -> i == 4.0, base_v2[:, 6]), [1,6]])
cluster5 = Matrix{Float64}(base_v2[findall(i -> i == 5.0, base_v2[:, 6]), [1,6]])
cluster6 = Matrix{Float64}(base_v2[findall(i -> i == 6.0, base_v2[:, 6]), [1,6]])
cluster7 = Matrix{Float64}(base_v2[findall(i -> i == 7.0, base_v2[:, 6]), [1,6]])


mc_c0, qmc_c0, c0_pvalue, c0_prop = compare_dist(cluster0[:, 1],Vector{Float64}(base_v2[:, 6]), sample, 0.0, "v2", n)
mc_c1, qmc_c1, c1_pvalue, c1_prop= compare_dist(cluster1[:, 1],Vector{Float64}(base_v2[:, 6]), sample, 1.0,"v2", n)
mc_c2, qmc_c2, c2_pvalue, c2_prop = compare_dist(cluster2[:, 1],Vector{Float64}(base_v2[:, 6]), sample, 2.0, "v2", n)
mc_c3, qmc_c3, c3_pvalue, c3_prop = compare_dist(cluster3[:, 1],Vector{Float64}(base_v2[:, 6]), sample, 3.0,"v2", n)
mc_c4, qmc_c4, c4_pvalue, c4_prop = compare_dist(cluster4[:, 1],Vector{Float64}(base_v2[:, 6]), sample, 4.0, "v2", n)
mc_c5, qmc_c5, c5_pvalue, c5_prop = compare_dist(cluster5[:, 1],Vector{Float64}(base_v2[:, 6]), sample, 5.0,"v2", n)
mc_c6, qmc_c6, c6_pvalue, c6_prop = compare_dist(cluster6[:, 1],Vector{Float64}(base_v2[:, 6]), sample, 6.0, "v2", n)
mc_c7, qmc_c7, c7_pvalue, c7_prop= compare_dist(cluster7[:, 1],Vector{Float64}(base_v2[:, 6]), sample, 7.0,"v2", n)

#Checking if the distributions of each cluster differ from other using KS-test
cst = Vector{Float64}(collect(0:7))
p_value = Matrix{Any}(zeros(9,9))
p_value[1,2:end] = ["Cluster 0","Cluster 1","Cluster 2","Cluster 3","Cluster 4","Cluster 5","Cluster 6","Cluster 7"]
p_value[2:end,1] = ["Cluster 0","Cluster 1","Cluster 2","Cluster 3","Cluster 4","Cluster 5","Cluster 6","Cluster 7"]

m = 0
n = 0
for i in cst
    m +=1
    for j in cst
        n+=1
        x = base_v2[findall(k -> k == i, base_v2[:, 6]), [1]][:, 1]
        y = base_v2[findall(k -> k == j, base_v2[:, 6]), [1]][:, 1]
        p_value[m+1,n+1] = pvalue(ApproximateTwoSampleKSTest(x, y))
    end
    n = 0
end

CSV.write("Databases/KS_test_clustering.csv", DataFrame(p_value, :auto))
