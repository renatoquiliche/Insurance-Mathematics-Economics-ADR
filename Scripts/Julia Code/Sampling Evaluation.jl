function sampling_evaluation(SampleGrid::Vector{Int64}, Master::DataFrame, Cluster::String,
    j)

    """
    println("Complete sample size: ", SampleGrid[j])
    """

    # Get the random, and quasi-random sequences
    # Random Numbers
    #Random.seed!(seed_mc[j])
    MC_idx = rand(Uniform(0,1), SampleGrid[j])
    #Random.seed!(seed_qmc[j])
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
    mc_index_unsorted = Vector{Int64}()
    qmc_index_unsorted = Vector{Int64}()

    # For MC bootstrap
    for i in 1:SampleGrid[j]
        mc_index_unsorted = vcat(mc_index_unsorted, findall(m -> m == MC_sample[i], Master[!, "charges"])[1])
        # [1] to get only the first value, original variable have some duplicates,
        # [1] after findall avoid picking repeated samples
    end

    # For QMC-LHS bootstrap
    for i in 1:SampleGrid[j]
        qmc_index_unsorted = vcat(qmc_index_unsorted, findall(m -> m == QMC_LHS_sample[i], Master[!, "charges"])[1])
        # [1] to get only the first value, original variable have some duplicates,
        # [1] after findall avoid picking repeated samples
    end

    # Subset the vector in the indexes
    """
    # Proportions of clusters in sample
    mean(Master[qmc_index_unsorted, Cluster] .== 1.00) # Re-sample
    mean(Master[qmc_index_unsorted, Cluster] .== 0.00) # Complete sample
    """

    # Re-sampled and sampled random variable

    MC_y = Master[qmc_index_unsorted, "charges"] # Re-sample
    QMC_y = Master[qmc_index_unsorted, "charges"] # Re-sample
    y = Master[!, "charges"] # Complete sample

    """
    println("For entire sample, \n",
            "The mean for bootstrap sample ", "j=", SampleGrid[j], mean(Master[qmc_index_unsorted, Cluster] .== 1.00),
            "\nThe true mean: ", mean(Master[!, Cluster] .== 1.00) )

    println("For entire sample, \n",
            "The mean for bootstrap sample ", "j=", SampleGrid[j], mean(Master[qmc_index_unsorted, Cluster] .== 0.00),
            "\nThe true mean: ", mean(Master[!, Cluster] .== 0.00) )
    """

    # Comparar la media de la simulacion y los datos reales, para cada cluster
    # To perform evaluations in each cluster,
    # it is necessary to find the indexes of clusters within the bootstrap samples

    """ Getting the samples """

    """ For Monte Carlo """
    k1_mc = findall(k -> k == 1.00, Master[mc_index_unsorted, Cluster])
    k0_mc = findall(k -> k == 0.00, Master[mc_index_unsorted, Cluster])

    k1_MC_y = Master[mc_index_unsorted, "charges"][k1_mc] # k1 Re-sampled
    k0_MC_y = Master[mc_index_unsorted, "charges"][k0_mc] # k0 Re-sampled

    """ For QMC-LHS """
    k1_qmc = findall(k -> k == 1.00, Master[qmc_index_unsorted, Cluster])
    k0_qmc = findall(k -> k == 0.00, Master[qmc_index_unsorted, Cluster])

    k1_QMC_y = Master[qmc_index_unsorted, "charges"][k1_qmc] # k1 Re-sampled
    k0_QMC_y = Master[qmc_index_unsorted, "charges"][k0_qmc] # k0 Re-sampled

    # For the original sample
    k1_y = Master[findall(k -> k == 1.00, Master[!, Cluster]), "charges"] # k1 Complete sample
    k0_y = Master[findall(k -> k == 0.00, Master[!, Cluster]), "charges"] # k0 Complete sample

    """ Mean t-test incorpored implementation """
    test_qmc = UnequalVarianceTTest(QMC_y, y) # Complete sample Re-sampled vs sampled
    test_mc = UnequalVarianceTTest(MC_y, y) # Complete sample Re-sampled vs sampled

    testk1_qmc = UnequalVarianceTTest(k1_QMC_y, k1_y) # Re-sampled vs sampled K1
    testk0_qmc = UnequalVarianceTTest(k0_QMC_y, k0_y) # Re-sampled vs sampled K0

    testk1_mc = UnequalVarianceTTest(k1_MC_y, k1_y) # Re-sampled vs sampled K1
    testk0_mc = UnequalVarianceTTest(k0_MC_y, k0_y) # Re-sampled vs sampled K0

    # println(test.xbar, "\n", test1.xbar, "\n", test0.xbar)

    gap_mean_qmc = test_qmc.xbar/mean(y)
    gap_mean_mc = test_mc.xbar/mean(y)

    gap_meank1_qmc = testk1_qmc.xbar/mean(y)
    gap_meank0_qmc = testk0_qmc.xbar/mean(y)

    gap_meank1_mc = testk1_mc.xbar/mean(y)
    gap_meank0_mc = testk0_mc.xbar/mean(y)

    # println("----------------------------------------------------")
    # println(round(gap_mean*100, digits=2), "%")

    # println("\n*****************************************************")
    resultsMC = Dict{String, Any}("Random Vector" => MC_y)
    resultsMC["Random Vector K1"] = k1_MC_y
    resultsMC["Random Vector K0"] = k0_MC_y
    resultsMC["T-test Complete Sample"] = test_mc
    resultsMC["T-test K1"] = testk1_mc
    resultsMC["T-test K0"] = testk0_mc
    resultsMC["Sample size"] = SampleGrid[j]

    resultsQMCLHS = Dict{String, Any}("Random Vector" => QMC_y)
    resultsQMCLHS["Random Vector K1"] = k1_QMC_y
    resultsQMCLHS["Random Vector K0"] = k0_QMC_y
    resultsQMCLHS["T-test Complete Sample"] = test_qmc
    resultsQMCLHS["T-test K1"] = testk1_qmc
    resultsQMCLHS["T-test K0"] = testk0_qmc
    resultsQMCLHS["Sample size"] = SampleGrid[j]

    #resultsMC["Sample size"] = Vector{Int64}(zeros(length(SampleGrid)))
    #resultsQMCLHS["Sample size"] = Vector{Int64}(zeros(length(SampleGrid)))

    return resultsMC, resultsQMCLHS

    # DataFrame([QMC_y, k1_kmeans_QMC_y, k0_kmeans_QMC_y, k1_kmeans_y, k0_kmeans_y], ["Random Vector", "a2", "a3", "a4", "a5"])
end
