function CDF(random_variable::Vector{Float64})
    Ordered_y = sort(random_variable)
    F_y = zeros(length(Ordered_y))
    for i in 1:1338
        F_y[i] = i/1338
    end
    Histogram_y = histogram(random_variable, label="Charges")
    return Ordered_y, F_y, Histogram_y
end
