
function print_local_convergence_rates(result)
    niterations = result.iterations
    niterations < 2 ? (println("Not enough iterations for computing the convergence rate."); return) : nothing
    fnorm(state) = state.fnorm
    fnorms = fnorm.(result.trace.states)
    convergence_rates = Float64[]
    for i = 1:niterations-1
        push!(convergence_rates, (log(fnorms[i+1])-log(fnorms[i+2]))/(log(fnorms[i])-log(fnorms[i+1])))
    end
    println("convergence rates: ", convergence_rates)
    return nothing
end

function get_convergence_rates(η::Vector{Float64})
    convergence_rates = Vector{Float64}(undef, length(η))
    for i=1:length(η)
        if i <= 2
            convergence_rates[i] = NaN
        else
            cr = (log(η[i])-log(η[i-1]))/(log(η[i-1])-log(η[i-2]))
            convergence_rates[i] = cr
        end
    end
    return convergence_rates
end

print_residuals(io, residuals::Vector) = print_residuals(io, Dict(:f=>residuals), false)

function print_residuals(io, residuals::Dict{Symbol, Vector{Float64}}, show_fieldnames=true)
    fields = collect(keys(residuals))
    niter = length(residuals[first(fields)])
    data = Matrix{Float64}(undef, niter, 1+2*length(fields))
    main_header = ["iter"]
    sub_header = [""]
    data[:,1] = 1:niter
    for (i, field) in enumerate(fields)
        data[:,2i] = residuals[field]
        data[:,2i+1] = get_convergence_rates(residuals[field])
        push!(main_header, "residual")
        push!(main_header, "rate")
        append!(sub_header, [string(field), string(field)])
    end
    header = show_fieldnames ? (main_header, sub_header) : main_header
    formatters = (ft_printf("%d", [1]), ft_printf("%5.3e", 2:size(data, 2)))
    highlighters = hl_col(3:2:size(data,2), crayon"dark_gray")
    pretty_table(io, data; header, formatters, highlighters)
end