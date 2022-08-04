using Random, Distributions,  XLSX, DataFrames, Plots,  CSV, QuasiMonteCarlo

base = XLSX.readdata("Database.xlsx", "Sheet1!B2:N1339")
dados = Matrix{Float64}(base[:, [4, end]])

ordered_y = sort(dados[:, 1])

Fy = zeros(length(ordered_y))
for i in 1:1338 
    Fy[i] = i/1338
end

plot(ordered_y, Fy)

mc_mean_std = zeros(20)
qmc_mean_std_unif = zeros(20)
qmc_mean_std_sobol = zeros(20)
qmc_mean_std_latin = zeros(20)
qmc_mean_std_lattice = zeros(20)

n = collect(500:500:10000)

for j in 1:20
    println("j = $j")
    MC_sample = zeros(n[j], 10)
    QMC_sample_unif = zeros(n[j], 10)
    QMC_sample_sobol = zeros(n[j], 10)
    QMC_sample_latin = zeros(n[j], 10)
    QMC_sample_lattice = zeros(n[j], 10)

   for m in 1:10 
        MC_idx = rand(Uniform(minimum(Fy),1), n[j])
        QMC_idx_unif = QuasiMonteCarlo.sample(n[j],minimum(Fy),1,UniformSample())
        QMC_idx_sobol = QuasiMonteCarlo.sample(n[j],minimum(Fy),1,SobolSample())
        QMC_idx_latin = QuasiMonteCarlo.sample(n[j],minimum(Fy),1,LatinHypercubeSample())
        QMC_idx_lattice = QuasiMonteCarlo.sample(n[j],minimum(Fy),1,LatticeRuleSample())

        for i in 1:n[j]
            mc_idx = maximum(findall(k -> k <= MC_idx[i], Fy))
            qmc_idx_unif = maximum(findall(k -> k <= QMC_idx_unif[i], Fy))
            qmc_idx_sobol = maximum(findall(k -> k <= QMC_idx_sobol[i], Fy))
            qmc_idx_latin = maximum(findall(k -> k <= QMC_idx_latin[i], Fy))
            qmc_idx_lattice = maximum(findall(k -> k <= QMC_idx_lattice[i], Fy))

            MC_sample[i,m] = ordered_y[mc_idx]
            QMC_sample_unif[i,m] = ordered_y[qmc_idx_unif]
            QMC_sample_sobol[i,m] = ordered_y[qmc_idx_sobol]
            QMC_sample_latin[i,m] = ordered_y[qmc_idx_latin]
            QMC_sample_lattice[i,m] = ordered_y[qmc_idx_lattice]
            #println(i)
        end
        println(m)
    end

    mc_mean_std[j] = mean(var.(eachcol(MC_sample)))
    qmc_mean_std_unif[j] = mean(var.(eachcol(QMC_sample_unif)))
    qmc_mean_std_sobol[j] = mean(var.(eachcol(QMC_sample_sobol)))
    qmc_mean_std_latin[j] = mean(var.(eachcol(QMC_sample_latin)))
    qmc_mean_std_lattice[j] = mean(var.(eachcol(QMC_sample_lattice)))
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