using Random, Distributions,  XLSX, DataFrames, Plots,  CSV, QuasiMonteCarlo, HypothesisTests

# Contracts based on clusters
KMeans = CSV.read("Databases/Database.csv", DataFrame)

# Contracts based on categories
Categories = CSV.read("Databases/contracts.csv", DataFrame)


Cluster_kmeans2 = hcat(KMeans.charges, KMeans.Cluster2)
Cluster_kmeans4 = hcat(KMeans.charges, KMeans.Cluster4)

Cluster_smoker = hcat(Categories.charges, Categories.Cluster_smoker)
Cluster_smoker_bmi = hcat(Categories.charges, Categories.Cluster_smoker_bmi)

Charges = Vector{Float64}(Categories.charges)

include("CDF.jl")

# Where y is the dependent variable Charges
ordered_y, Fy, hist_charges = CDF(Charges)

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

j = 10

println("Sample size: ", SampleGrid[j])
MC_sample = zeros(SampleGrid[j])
MC_cluster_v0 = zeros(SampleGrid[j]) #cluster by algorithm
MC_cluster_v1 = zeros(SampleGrid[j]) #cluster by smoker variable
MC_cluster_v2 = zeros(SampleGrid[j]) #cluster by smoder, BMI and age variables

QMC_sample_latin = zeros(SampleGrid[j])
QMC_cluster_v0_latin = zeros(SampleGrid[j])
QMC_cluster_v1_latin = zeros(SampleGrid[j])
QMC_cluster_v2_latin = zeros(SampleGrid[j])

# Get the random, and quasi-random sequences
Random.seed!(seed_mc[j])
MC_idx = rand(Uniform(0,1), SampleGrid[j])
Random.seed!(seed_qmc[j])
QMC_idx_latin = QuasiMonteCarlo.sample(SampleGrid[j],0,1,LatinHypercubeSample(threading=true))

# Foreach i (point in sample for batch j), get the n[i] samples using MC and QMC, also get the clusters v0, v1 and v1 to which they belong
mc_idx = Vector{Int64}()
qmc_idx_latin = Vector{Int64}()

for i in 1:1000
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



sample["MC"]["value"][n[j]] = MC_sample
sample["MC"]["cluster"]["v0"][n[j]] = MC_cluster_v0
sample["MC"]["cluster"]["v1"][n[j]] = MC_cluster_v1
sample["MC"]["cluster"]["v2"][n[j]] = MC_cluster_v2
sample["QMC"]["value"][n[j]] = QMC_sample_latin
sample["QMC"]["cluster"]["v0"][n[j]] = QMC_cluster_v0_latin
sample["QMC"]["cluster"]["v1"][n[j]] = QMC_cluster_v1_latin
sample["QMC"]["cluster"]["v2"][n[j]] = QMC_cluster_v2_latin
end
