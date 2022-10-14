using Random, Distributions, XLSX, DataFrames, Plots, CSV, QuasiMonteCarlo, HypothesisTests, StatsBase

using PyCall
using PyPlot

ENV["PYTHON"] = "C:\\Users\\User\\AppData\\Local\\Programs\\Python\\Python310\\python.exe"
Pkg.build("PyCall")

Data = CSV.read("Databases/contracts.csv", DataFrame)

charges = Vector{Float64}(Data.charges)./1000

resample_size = 1000

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
    return vec(s), hist_s
end

MC_bootstrap(charges, 800)[2]


QMC_LHS_empirical(charges, 800)[2]


"""
Python code
"""
scipystats = pyimport("scipy.stats")

distribution_distance_MC = scipystats.wasserstein_distance(u_values=charges, v_values=charges_resampled_MC)
distribution_distance_QMC = scipystats.wasserstein_distance(u_values=charges, v_values=charges_resampled_QMC)

"""
end
"""


"""
Performance evaluation of re-sampling
"""
resample_grid = collect(100:50:2000)
replications = 100

distribution_distance_MC = zeros(replications, length(resample_grid))
distribution_distance_QMC = zeros(replications, length(resample_grid))
# Rows = Replications
# Columns = variations of sample size

Random.seed!(0)
for i in 1:replications
    for j in 1:length(resample_grid)
        charges_resampled_MC = MC_bootstrap(charges, resample_grid[j])[1]
        charges_resampled_QMC = QMC_LHS_empirical(charges, resample_grid[j])[1]

        distribution_distance_MC[i,j] = scipystats.wasserstein_distance(u_values=charges, v_values=charges_resampled_MC)
        distribution_distance_QMC[i,j] = scipystats.wasserstein_distance(u_values=charges, v_values=charges_resampled_QMC)
    end
end

histogram(distribution_distance_MC[:,length(resample_grid)], bins=:sqrt)
histogram(distribution_distance_QMC[:,length(resample_grid)], bins=:sqrt)

println("------------------------------------------------------")
println("Sample size: ", resample_grid[length(resample_grid)], " mean of MC distance from original distribution: ",
mean(distribution_distance_MC[:,length(resample_grid)]))


println("Sample size: ", resample_grid[length(resample_grid)], " mean of QMC distance from original distribution: ",
mean(distribution_distance_QMC[:,length(resample_grid)]))
println("------------------------------------------------------")

mean(distribution_distance_MC[:,:])

avg_MC = mean.(eachcol(distribution_distance_MC))
sd_MC = std.(eachcol(distribution_distance_MC))

avg_QMC = mean.(eachcol(distribution_distance_QMC))
sd_QMC = std.(eachcol(distribution_distance_QMC))


visualization = 0.02

figure(figsize=(12, 7))
ylabel("Wassertain Distance over $replications replications", fontsize=15)
xlabel("Number of scenarios (resample size)", fontsize=15)
title("Convergence plot MC", fontsize=20)
PyPlot.plot(resample_grid, avg_MC, color="tab:blue", linewidth=2.0, label="MC Bootstrap")
PyPlot.errorbar(resample_grid, avg_MC, yerr=sd_MC, fmt="o")
PyPlot.grid("on")
#PyPlot.fill_between(resample_grid, avg_MC.-sd_MC, avg_MC.+sd_MC, alpha=0.5, linestyle="dashed", edgecolor="k", linewidth=2, antialiased=true)
#PyPlot.savefig("Distance from original distribution MC.png")

#figure(figsize=(12, 7))
#ylabel("Mean Wassertain Distance over $replications replications", fontsize=15)
#xlabel("Number of scenarios (resample size)", fontsize=15)
#title("Convergence plot QMC", fontsize=20)
PyPlot.plot(resample_grid, avg_QMC, color="tab:orange", linewidth=2.0, label="QMC-LHS")
PyPlot.errorbar(resample_grid, avg_QMC, yerr=sd_QMC.+visualization, fmt="o")
PyPlot.legend()
#PyPlot.fill_between(resample_grid, avg_QMC.-sd_QMC, avg_QMC.+sd_QMC, alpha=0.5, linestyle="dashed", edgecolor="k", linewidth=2, antialiased=true)
PyPlot.savefig("Results\\Distance from original distribution.png")
