using Random, Distributions, XLSX, DataFrames, Plots, CSV, QuasiMonteCarlo, HypothesisTests, StatsBase

Data = CSV.read("Databases/contracts.csv", DataFrame)

charges = Vector{Float64}(Data.charges)

function MC_bootstrap(y, sample_size)
    x = ones(sample_size,1)

    StatsBase.sample!(y, x, replace=true, ordered=false)
    hist_x = histogram(y, label="Sampled", bins=:sqrt)
    hist_x = histogram!(x, label="Re-sampled MC", fillalpha=0.7, bins=:sqrt)
    return x, hist_x
end

charges_resampled_MC, h_charges_resampled_MC = MC_bootstrap(charges, 800)

h_charges_resampled_MC

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

charges_resampled_QMC, h_charges_resampled_QMC = QMC_LHS_empirical(charges, 1338)

h_charges_resampled_QMC

using KDEstimation, Distributions
# set a seed for reproducibility
using Random: seed!
seed!(1234)
# generate random data
x = randn(1000)
rot = rule_of_thumb2(Normal,x)
println("rule of thumb: ", rot)
lscv_res = lscv(Normal,x,FFT())

plot(lscv_res)

h = minimizer(lscv_res)
fkde = kde(Normal, h, x, FFT())
frot = kde(Normal, rot, x, FFT())

plot(fkde, label="LSCV", lw=2)
plot!(frot, label="Rule of thumb", lw=2)

# KDE -> probabilities
function quad_midpoint(f,a,b,N)
    h = (b-a)/N
    int = 0.0
    for k=1:N
        xk_mid = (b-a) * (2k-1)/(2N) + a
        int = int + h*f(xk_mid)
    end
    return int
end

quad_midpoint(fkde,-0.5,0.5,1000)

function KLD1(x)
    term = fkde(x)*log(frot(x)/fkde(x))
    return term
end

KLD = quad_midpoint(KLD1, -0.5, 0.5, 1000)
