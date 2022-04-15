abstract type AbstractGuess end

"""
    NoGuess()

Take the last converged time steps solution as initial guess for the next time step.

This is the simples initial guess, often not the best one.
"""
struct NoGuess <: AbstractGuess end

"""
    FunctionBasedGuess(f::Function)

Take an intial guess based on the last two converged time steps solutions that is scaled by
a time-dependent scalar function f,
i.e for previous solutions ⁿ⁻²u and ⁿ⁻¹u, and associated
times ⁿ⁻²t, ⁿ⁻¹t, ⁿt , the guess for the current solution ⁿu is:
ⁿu = ⁿ⁻¹u + (ⁿ⁻¹u - ⁿ⁻²u)*(f(ⁿt) - f(ⁿ⁻¹t))/(f(ⁿ⁻¹t) - f(ⁿ⁻²t))
.

This can be a good choice for problems whose behavior is dominated by loading curves,
e.g. cyclic loading. In that case a scalar function representing the loading should be used
for constructing the guess.
"""
struct FunctionBasedGuess{F} <: AbstractGuess
    f::F
end

"""
    TimeBasedGuess()

Take an intial guess based on the last two converged time steps solutions that is scaled by time,
i.e for previous solutions ⁿ⁻²u and ⁿ⁻¹u, and associated
times ⁿ⁻²t, ⁿ⁻¹t, ⁿt , the guess for the current solution ⁿu is:
ⁿu = ⁿ⁻¹u + (ⁿ⁻¹u - ⁿ⁻²u)*(ⁿt - ⁿ⁻¹t)/(ⁿ⁻¹t - ⁿ⁻²t)
.
This can be a good choice for time-dependent problems.
"""
struct TimeBasedGuess <: AbstractGuess end

# columns of var: 1: n-2, 2: n-1, 3: n (=current step)
function initial_guess!(vars::Matrix{Float64}, ::NoGuess, ts)
    # copy!(@view(vars[:,end]), vars[:,end-1])
    @views copy!(vars[:,end], vars[:,end-1])
    return vars
end

function initial_guess!(vars::Matrix{Float64}, guess::FunctionBasedGuess, ts)
    initial_guess!(vars, guess.f, ts)
end

function initial_guess!(vars::Matrix{Float64}, guess::TimeBasedGuess, ts)
    initial_guess!(vars, identity, ts)
end

function initial_guess!(vars::Matrix{Float64}, f::Function, ts)
    for (i, v) in enumerate(eachrow(vars))
        vars[i,end] = function_based_guess(f, v, ts)
    end
    return vars 
end

# compute the actual guess
function function_based_guess(f, v::AbstractVector{Float64}, ts::AbstractVector{Float64})
    if f(ts[end-1]) ≈ f(ts[end-2])
        scaling = 0.0
    else
        scaling = (f(ts[end]) - f(ts[end-1])) / (f(ts[end-1]) - f(ts[end-2]))
    end
    v_guess = v[end-1] + (v[end-1] - v[end-2]) * scaling
    return v_guess
end