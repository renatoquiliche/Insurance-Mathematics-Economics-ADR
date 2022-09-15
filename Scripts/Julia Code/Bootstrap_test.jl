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

include("Sampling Evaluation.jl")
# Estratificado por clusters
# Proporciones ta bom
# Medias, no esta tan bom

# Create the dictionaries to store results
resMC = Dict()
resQMC = Dict()
for i in 1:30
    for j in 1:length(SampleGrid)
        Random.seed!(i+j)
        resMC[i,j], resQMC[i,j] = sampling_evaluation(SampleGrid, Master, "Cluster2", j)
    end
end

# Replication i, sample size j
Replications = 30
bootstrap_eval = zeros(Replications)

for i in 1:Replications
    bootstrap_eval[i] = resMC[i, 20]["T-test Complete Sample"].xbar
end

mean(bootstrap_eval), std(bootstrap_eval)

resMC[1,1]["T-test Complete Sample"].xbar
resMC[2,1]["T-test Complete Sample"].xbar

## Evaluate the convergence


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
# Incorporar las seeds en las simulaciones
