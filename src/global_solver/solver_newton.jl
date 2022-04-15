struct Newton{T, R}
    max_iterations::Int
    max_nincreases::Int
    # tolerance scaling
    tolerance::T
    residuals::R
end

function Newton(max_iterations::Int, max_nincreases::Int, t::AbstractTolerance)
    residuals = get_residuals(t)
    return Newton(max_iterations, max_nincreases, t, residuals)
end

function reinit_solver!(s::Newton{T, Vector{Float64}}) where T
    deleteat!(s.residuals, 1:length(s.residuals))
    return s
end

function reinit_solver!(s::Newton{T, Dict{Symbol, Vector{Float64}}}) where T
    for key in keys(s.residuals)
        s.residuals[key] = Float64[]
    end
    return s
end

function nincreases(residuals::Vector{Float64})
    length(residuals) == 1 && return 0
    n_increases = 0
    for i = reverse(2:length(residuals))
        if residuals[i] > residuals[i-1]
            n_increases += 1
        end
    end
    return n_increases
end

function nincreases(residuals::Dict{Symbol, Vector{Float64}})
    n_increases = Int[]
    for (name, res) in residuals
        push!(n_increases, nincreases(res))
    end
    return maximum(n_increases)
end

struct GlobalConvergenceError <: Exception
    residuals
end

function Base.showerror(io::IO, e::GlobalConvergenceError)
    println(io, "No convergence from global solver. Last residuals:")
    print_residuals(io, e.residuals)
end

# find solution for current time step by Newton-Raphson iterations
# global solver options:
# max_iterations: maximum number of Newton-Raphson iterations
# tolerances: global tolerance per field as a dictionary
function iterate!(
    vars::AbstractMatrix{Float64},
    solver::Newton,
    fe_sets::Vector{<:FESet},
    assembler,
    dh::MixedDofHandler,
    ch::ConstraintHandler,
    nbcs::Tuple = (), # Neumann boundary conditions
    local_solver_options=Dict{Symbol, Any}(),
    Δt::Float64 = 1.0,
)

    # all assemblers are based on one and the same K and f
    K = Ferrite.getsparsemat(assembler)
    f = getvector(assembler)

    converged = false
    reinit_solver!(solver)
    for iter = 1:solver.max_iterations
        fill!(K.nzval, 0.0); fill!(f, 0.0)
        @views assembly!(assembler, fe_sets, dh, vars[:, 2:3], nbcs, local_solver_options, Δt)
        # tolerance scaling
        iter == 1 && update_tolerance!(solver.tolerance, f)
        compute_residuals!(solver.residuals, f, ch, solver.tolerance)
        converged = check_convergence(solver.residuals, solver.tolerance)
        
        if converged
            @views copy!(vars[:,1], vars[:,2])
            @views copy!(vars[:,2], vars[:,3])
            # save converged material state
            for feset in fe_sets
                update_material_states!(feset)
            end
            break
        else
            if iter == solver.max_iterations || nincreases(solver.residuals) > solver.max_nincreases
                throw(GlobalConvergenceError(solver.residuals))
            end
        end
        apply_zero!(K, f, ch)
        vars[:, 3] -=  K \ f
    end
    return converged
end

function compute_residuals!(
    residuals::Dict{Symbol, Vector{T}},
    f::AbstractVector{T}, 
    ch::ConstraintHandler,
    t::AbstractTolerance,
    ) where {T}

    for (fieldname, dofs) in t.fielddofs
        @views residual = norm(f[intersect(dofs, free_dofs(ch))])
        push!(residuals[fieldname], residual)
    end
    return residuals
end

function compute_residuals!(
    residuals::Vector{T},
    f::AbstractVector{T},
    ch::ConstraintHandler,
    t::AbstractTolerance,
) where {T}

    @views residual = norm(f[free_dofs(ch)])
    push!(residuals, residual)
    return residuals
end
