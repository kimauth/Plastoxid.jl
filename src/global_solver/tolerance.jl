abstract type AbstractToleranceScaling end

"""
    AbsoluteTolerance(tol::Float64)

Represents no tolerance scaling, i.e. `tol` is used as tolerance.
"""
struct AbsoluteTolerance <: AbstractToleranceScaling
    tol::Float64
end

scaled_tolerance(ts::AbsoluteTolerance, ::AbstractVector) = ts.tol

"""
    ReactionScaling(base_tol::Float64, min_tol::Float64)

Represents tolerance scaling based on the residual vector (including entries relating to 
Dirichlet boundary conditions, thus `ReactionScaling`) in the first iteration.
The tolerance for a residual vector `r` is determined such that
tol = max(min_tol, base_tol*norm(r)).
"""
struct ReactionScaling <: AbstractToleranceScaling
    base_tol::Float64
    min_tol::Float64
end

function scaled_tolerance(ts::ReactionScaling, r::AbstractVector{T}) where T
    reaction = norm(r)
    tol = max(ts.min_tol, ts.base_tol*reaction)
    return tol
end

abstract type AbstractTolerance end

mutable struct GlobalTolerance{TS} <: AbstractTolerance
    tol::Float64
    scaling::TS
    GlobalTolerance(s::TS) where TS<:AbstractToleranceScaling = new{TS}(0.0, s) 
end

"""
    check_convergence(residuals, t::AbstractTolerance) where T 

Check if the last residual in `residuals` fullfills the convergence criterion given by `t`.
"""
function check_convergence(
    residuals::Vector{T},
    t::GlobalTolerance,
    ) where {T}
    residual = last(residuals)
    if residual > t.tol
        return false
    else
        return true
    end
end

function update_tolerance!(t::GlobalTolerance, f::AbstractVector{T}) where T
    @views tol = scaled_tolerance(t.scaling, f)
    t.tol = tol
end

get_residuals(::GlobalTolerance) = Float64[]

"""
    FieldTolerance(scaling::Dict{Symbol, AbstractToleranceScaling})

Tolerance scaling per field. `scaling` must hold a `AbstractToleranceScaling` for each field
with the Symbol representing the field name as key.

Available tolerance scaling options are:
- [`AbsoluteTolerance`](@ref)
- [`ReactionScaling`](@ref)
"""
struct FieldTolerance <: AbstractTolerance
    tol::Dict{Symbol, Float64}
    scaling::Dict{Symbol, <:AbstractToleranceScaling}
    fielddofs::Dict{Symbol, Vector{Int}}

    function FieldTolerance(
        scaling::Dict{Symbol, <:AbstractToleranceScaling},
        fielddofs::Dict{Symbol, Vector{Int}}
    )
        tol = Dict{Symbol, Float64}(name => 0.0 for name in keys(scaling))
        all(keys(scaling) .== keys(fielddofs)) || error("Fieldnames in scaling and fielddofs are not the same.")
        return new(tol, scaling, fielddofs)
    end
end

function check_convergence(
    residuals::Dict{Symbol, Vector{T}},
    t::FieldTolerance,
    ) where {T}

    for (fieldname, residual_vector) in residuals
        residual = last(residual_vector)
        tolerance = t.tol[fieldname]
        if residual > tolerance
            return false
        end
    end
    return true
end

function update_tolerance!(
    t::FieldTolerance,
    f::AbstractVector{T},
) where T
    for (fieldname, dofs) in t.fielddofs
        @views tol = scaled_tolerance(t.scaling[fieldname], f[dofs])
        t.tol[fieldname] = tol
    end
end

get_residuals(t::FieldTolerance) = Dict(key => Float64[] for key in keys(t.tol))
    
