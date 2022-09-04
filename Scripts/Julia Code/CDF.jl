function CDF(random_variable::Vector{Float64})
    ordered_y = sort(random_variable)
    Fy = zeros(length(ordered_y))
    for i in 1:1338
        Fy[i] = i/1338
    end
    hist = histogram(random_variable, label="Charges")
    return ordered_y, Fy, hist
end
