using Random, Distributions,  XLSX, DataFrames, Plots,  CSV, QuasiMonteCarlo, HypothesisTests, IJulia, StatsBase

# Contracts based on clusters
KMeans = CSV.read("Databases/Database.csv", DataFrame)

# Contracts based on categories
Categories = CSV.read("Databases/contracts.csv", DataFrame)

Charges = Vector{Float64}(Categories.charges)

Master = hcat(KMeans[!, ["charges", "Cluster2", "Cluster4"]],
                Categories[!, ["Cluster_smoker", "Cluster_smoker_bmi"]])

include("CDF.jl")

Master.charges

# Where y is the dependent variable Charges
ordered_y, Fy, hist_charges = CDF(Master[!, "charges"])

# Define the Dicts to store the simulations
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

#Grid of the sample sizes to analyze simulations
SampleGrid = collect(100:100:2000)

# The seeds for each sample size within the Grid,
# Using the same seed will lead to bigger samples, but with same indexes
# The objective is to found a point of convergence

# Random seed for sample k of size n_j
Random.seed!(123)
seed_mc = rand(collect(1:1000), length(SampleGrid)) #seeds used in the Monte carlo sampling
Random.seed!(321)
seed_qmc = rand(collect(1:1000), length(SampleGrid)) #seeds used in the Quasi-Monte carlo sampling

function sampling_evaluation(SampleGrid::Vector{Int64}, Master::DataFrame, Cluster::String)
    for j in 1:length(SampleGrid)

        """
        println("Complete sample size: ", SampleGrid[j])
        """

        # Get the random, and quasi-random sequences
        # Random Numbers
        Random.seed!(seed_mc[j])
        MC_idx = rand(Uniform(0,1), SampleGrid[j])
        Random.seed!(seed_qmc[j])
        QMC_idx_latin = QuasiMonteCarlo.sample(SampleGrid[j],0,1,LatinHypercubeSample(threading=true))

        # Foreach i (point in sample for batch j), get the n[i] samples using MC and QMC, also get the clusters v0, v1 and v1 to which they belong
        mc_idx = Vector{Int64}()
        qmc_idx_latin = Vector{Int64}()

        for i in 1:SampleGrid[j]
            #obtaining the index of sampled value in the real cumulative distribution Random Bootstrap
            if MC_idx[i] < Fy[1]
                mc_idx = push!(mc_idx, 1)
            else
                mc_idx = push!(mc_idx, maximum(findall(k -> k <= MC_idx[i], Fy)))
            end

            #obtaining the index of sampled value in the real cumulative distribution Quasi-Random Bootstrap - LHS
            if QMC_idx_latin[i] < Fy[1]
                qmc_idx_latin = push!(qmc_idx_latin, 1)
            else
                qmc_idx_latin = push!(qmc_idx_latin, maximum(findall(k -> k <= QMC_idx_latin[i], Fy)))
            end
        end

        """
        println("Random Bootstrap (Monte Carlo) unique samples: ", length(unique(mc_idx)) )
        println("Quasi-Monte Carlo LHS unique samples: ", length(unique(qmc_idx_latin)) )
        """

        MC_sample = ordered_y[mc_idx]
        QMC_LHS_sample = ordered_y[qmc_idx_latin]

        # Get Cluster indexes for each Cluster KMeans, Categorical (smoker) and Categorical (smoker + bmi)
        # This method uses the obtained sampled values for MC and QMC-lHS

        # This step is necessary because the sampled indexes for MC and QMC-LHS are valid for sortered random vector
        # This method matches by values and get the indexes on the unsorted random vector,
        # that will be used to further analysis
        Cat_smoker = Vector{Int64}()
        for i in 1:SampleGrid[j]
            Cat_smoker = vcat(Cat_smoker, findall(m -> m == QMC_LHS_sample[i], Master[!, "charges"])[1])
            # [1] to get only the first value, original variable have some duplicates,
            # [1] after findall avoid picking repeated samples
        end

        # Subset the vector in the indexes

        # Proportions of clusters in sample
        mean(Master[Cat_smoker, Cluster] .== 1.00) # Re-sample
        mean(Master[Cat_smoker, Cluster] .== 0.00) # Complete sample

        # Re-sampled and sampled random variable

        QMC_y = Master[Cat_smoker, "charges"] # Re-sample
        y = Master[!, "charges"] # Complete sample

        """
        println("For entire sample, \n",
                "The mean for bootstrap sample ", "j=", SampleGrid[j], mean(Master[Cat_smoker, Cluster] .== 1.00),
                "\nThe true mean: ", mean(Master[!, Cluster] .== 1.00) )

        println("For entire sample, \n",
                "The mean for bootstrap sample ", "j=", SampleGrid[j], mean(Master[Cat_smoker, Cluster] .== 0.00),
                "\nThe true mean: ", mean(Master[!, Cluster] .== 0.00) )
        """

        # Comparar la media de la simulacion y los datos reales, para cada cluster
        # To perform evaluations in each cluster,
        # it is necessary to find the indexes of clusters within the bootstrap samples
        k1 = findall(k -> k == 1.00, Master[Cat_smoker, Cluster])
        k0 = findall(k -> k == 0.00, Master[Cat_smoker, Cluster])

        k1_kmeans_QMC_y = Master[Cat_smoker, "charges"][k1] # k1 Re-sampled
        k0_kmeans_QMC_y = Master[Cat_smoker, "charges"][k0] # k0 Re-sampled

        # For the original sample
        k1_kmeans_y = Master[findall(k -> k == 1.00, Master[!, Cluster]), "charges"] # k1 Complete sample
        k0_kmeans_y = Master[findall(k -> k == 0.00, Master[!, Cluster]), "charges"] # k0 Complete sample

        test = UnequalVarianceTTest(QMC_y, y)
        test1 = UnequalVarianceTTest(k1_kmeans_QMC_y, k1_kmeans_y)
        test0 = UnequalVarianceTTest(k0_kmeans_QMC_y, k0_kmeans_y)

        # println(test.xbar, "\n", test1.xbar, "\n", test0.xbar)

        gap_mean = test.xbar/mean(y)

        # println("----------------------------------------------------")
        # println(round(gap_mean*100, digits=2), "%")

        # println("\n*****************************************************")
    end
end

include("Sampling Evaluation.jl")
# Estratificado por clusters
# Proporciones ta bom
# Medias, no esta tan bom
for j in 1:length(SampleGrid)
    a = sampling_evaluation(collect(100:100:2000), Master, "Cluster2", 1)
end

a["Random Vector"]




## Add, the MC bootstrap, return objects for each vectors, the cluster indexing

Master

index = 1:length(Master.charges)

# J =
gap = zeros(length(100:100:2000),length(1:100))
for i in 100:100:2000
    for j in 1:100
        Random.seed!(j)
        MC_index = StatsBase.sample(1:nrow(Master), i, replace=true)
        gap[Int64(i/100),j] = abs(mean(Master[MC_index, "charges"]) - mean(Master[!, "charges"])) / mean(Master[!, "charges"])
    end
println("Bootstrap size = $i ", "mean: ", mean(gap[Int64(i/100), :])*100)
end

# Medir con MAPE, RMSE, distancia t-test, KLD

sample["MC"]["value"][n[j]] = MC_sample
sample["MC"]["cluster"]["v0"][n[j]] = MC_cluster_v0
sample["MC"]["cluster"]["v1"][n[j]] = MC_cluster_v1
sample["MC"]["cluster"]["v2"][n[j]] = MC_cluster_v2
sample["QMC"]["value"][n[j]] = QMC_sample_latin
sample["QMC"]["cluster"]["v0"][n[j]] = QMC_cluster_v0_latin
sample["QMC"]["cluster"]["v1"][n[j]] = QMC_cluster_v1_latin
sample["QMC"]["cluster"]["v2"][n[j]] = QMC_cluster_v2_latin
