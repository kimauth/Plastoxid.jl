struct CoupledKolluri <: AbstractMaterial
    D::Float64
    cᵍᵇ_char::Float64 # environmental oxygen concentration
    cᵍᵇ₀::Float64 # reference oxygen concentration
    Vₒ₂::Float64 # partial molar volume of oxygen in metal
    δₙ::Float64
    δₜ::Float64
    Φₙ::Float64
    Φₜ::Float64
    M::Float64 # pressure factor
    dₒ₂ₘₐₓ::Float64
    # not really material parameters
    R::Float64
    T::Float64
end

# TODO: perhaps change MaterialModels.jl syntax to this
Φₙ(::Type{CoupledKolluri}, σₘₐₓ::T, δₙ::T) where T = σₘₐₓ * δₙ / (one(T) - (one(T) - exp(-one(T))) * Plastoxid.ℋᵣ(δₙ, δₙ))
Φₜ(::Type{CoupledKolluri}, τₘₐₓ::T, δₜ::T) where T = τₘₐₓ * δₜ * exp(T(0.5))
σₘₐₓ(::Type{CoupledKolluri}, Φₙ::T, δₙ::T) where T = Φₙ / δₙ * (one(T) - (one(T) - exp(-one(T))) * Plastoxid.ℋᵣ(δₙ, δₙ))
τₘₐₓ(::Type{CoupledKolluri}, Φₜ::T, δₜ::T) where T = Φₜ / δₜ * exp(-T(0.5))

struct CoupledKolluriState{dim_s,T} <: AbstractMaterialState
    cᵍᵇₘₐₓ::T
    Δₘₐₓ::Vec{dim_s,T}
end

Base.zero(::Type{CoupledKolluriState{dim_s,T}}) where {dim_s,T} = CoupledKolluriState(zero(T), zero(Vec{dim_s, T}))
MaterialModels.initial_material_state(::CoupledKolluri, dim_s::Int = 2) = zero(CoupledKolluriState{dim_s, Float64})

MaterialModels.get_cache(::CoupledKolluri) = nothing 

# modified van den Bosch cohesive law with damage according to Kolluri et al.
function MaterialModels.material_response(
    material::CoupledKolluri,
    Δ::Vec{dim_s,TT},
    cᵍᵇ::TT,
    state::CoupledKolluriState{dim_s, TT},
    Δt=nothing;
    cache=nothing,
    options=nothing,
    ) where {dim_s, TT}

        Δₜ, Δₙ = split_jump(Δ)
        
        ∂cᵍᵇₘₐₓ∂cᵍᵇ, cᵍᵇₘₐₓ = gradient(cᵍᵇ -> max_concentration(cᵍᵇ, state.cᵍᵇₘₐₓ), cᵍᵇ, :all)
        ∂Δₘₐₓ∂Δ, Δₘₐₓ = gradient(Δ -> max_jump(Δ, state.Δₘₐₓ), Δ, :all)

        ∂T∂cᵍᵇ = gradient(cᵍᵇ -> T(material, Δ, Δₘₐₓ, cᵍᵇ, cᵍᵇₘₐₓ), cᵍᵇ)
        ∂T∂cᵍᵇₘₐₓ = gradient(cᵍᵇₘₐₓ -> T(material, Δ, Δₘₐₓ, cᵍᵇ, cᵍᵇₘₐₓ), cᵍᵇₘₐₓ)
        dTdcᵍᵇ = ∂T∂cᵍᵇ + ∂T∂cᵍᵇₘₐₓ * ∂cᵍᵇₘₐₓ∂cᵍᵇ

        ∂T∂Δ, _T = gradient(Δ -> T(material, Δ, Δₘₐₓ, cᵍᵇ, cᵍᵇₘₐₓ), Δ, :all)
        ∂T∂Δₘₐₓ = gradient(Δₘₐₓ -> T(material, Δ, Δₘₐₓ, cᵍᵇ, cᵍᵇₘₐₓ), Δₘₐₓ)
        dTdΔ = ∂T∂Δ + ∂T∂Δₘₐₓ ⋅ ∂Δₘₐₓ∂Δ

        # "extra" derivatives
        ∂²T̂ₙ∂cᵍᵇₘₐₓ², ∂T̂ₙ∂cᵍᵇₘₐₓ, _ = hessian(cᵍᵇₘₐₓ -> T̂ₙ(material, Δ, Δₘₐₓ, cᵍᵇₘₐₓ), cᵍᵇₘₐₓ, :all)
        ∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δₘₐₓ = gradient(Δₘₐₓ->gradient(cᵍᵇₘₐₓ -> T̂ₙ(material, Δ, Δₘₐₓ, cᵍᵇₘₐₓ), cᵍᵇₘₐₓ), Δₘₐₓ)
        ∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δ = gradient(Δ->gradient(cᵍᵇₘₐₓ -> T̂ₙ(material, Δ, Δₘₐₓ, cᵍᵇₘₐₓ), cᵍᵇₘₐₓ), Δ)
        ∂²T̂ₙ∂Δₘₐₓ², ∂T̂ₙ∂Δₘₐₓ, _ = hessian(Δₘₐₓ -> T̂ₙ(material, Δ, Δₘₐₓ, cᵍᵇₘₐₓ), Δₘₐₓ, :all)
        ∂²T̂ₙ∂Δₘₐₓ∂Δ = gradient(Δ->gradient(Δₘₐₓ -> T̂ₙ(material, Δ, Δₘₐₓ, cᵍᵇₘₐₓ), Δₘₐₓ), Δ)
        ∂²T̂ₙ∂Δ², ∂T̂ₙ∂Δ, _ = hessian(Δ -> T̂ₙ(material, Δ, Δₘₐₓ, cᵍᵇₘₐₓ), Δ, :all)

        # TODO: instead of returning the 2nd derivatives, we should return total derivatives
        #  d_∂T̂ₙ∂cᵍᵇₘₐₓ_dΔ, d_∂T̂ₙ∂cᵍᵇₘₐₓ_dc etc.
        extras = (;
            # 1st derivatives
            ∂T̂ₙ∂cᵍᵇₘₐₓ,
            ∂T̂ₙ∂Δₘₐₓ,
            ∂T̂ₙ∂Δ,
            # 2nd derivatives
            ∂²T̂ₙ∂cᵍᵇₘₐₓ²,
            ∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δₘₐₓ,
            ∂²T̂ₙ∂cᵍᵇₘₐₓ∂Δ,
            ∂²T̂ₙ∂Δₘₐₓ²,
            ∂²T̂ₙ∂Δₘₐₓ∂Δ,
            ∂²T̂ₙ∂Δ²,
            # history variable derivatives
            ∂Δₘₐₓ∂Δ,
            ∂cᵍᵇₘₐₓ∂cᵍᵇ,
            )

        return _T, dTdΔ, dTdcᵍᵇ, CoupledKolluriState(cᵍᵇₘₐₓ, Δₘₐₓ), extras
end

@inline ℋᵣ(x, δ) = 1/2 + atan(x / 0.00001δ) / π

@inline function split_jump(Δ::Vec{dim_s, T}) where {dim_s, T}
    Δₜ = Vec{dim_s-1, T}(i->Δ[i])
    Δₙ = Δ[end]
    return Δₜ, Δₙ
end

@inline function max_concentration(c::T1, c_max::T2) where {T1, T2} 
    T = promote_type(T1, T2)
    if c > c_max
        return T(c)
    else
        return T(c_max)
    end
end

# compute state variables for cohesive damage
@inline function max_jump(Δ::Vec{dim,T1}, Δₘₐₓ::Vec{dim,T2}) where {dim, T1, T2}

    function _max_jump_percomponent(Δ::Vec{dim, T1}, Δₘₐₓ::Vec{dim, T2}, i::Int) where {dim, T1, T2}
        T = promote_type(T1, T2)
        if i < dim 
            v = abs(Δ[i]) >= Δₘₐₓ[i] ? abs(Δ[i]) : Δₘₐₓ[i] # tangential components
            return T(v)
        else 
            v = Δ[i] >= Δₘₐₓ[i] ? Δ[i] : Δₘₐₓ[i] # normal component
            return T(v)
        end
    end

    return Vec{dim}(i->_max_jump_percomponent(Δ, Δₘₐₓ, i))
end

# damage variables
@inline d_n(m::CoupledKolluri, Δₙ_max) = one(Δₙ_max) - exp(-Δₙ_max / m.δₙ)
@inline d_ct(m::CoupledKolluri, Δₜ_max::Vec{dim,T}) where {dim,T} = one(T) - exp(-Δₜ_max ⋅ Δₜ_max / 2m.δₜ^2)
@inline d_t(m::CoupledKolluri, Δₜ_max::Vec{dim,T}) where {dim,T} = one(T) - exp(-Δₜ_max ⋅ Δₜ_max / 2m.δₜ^2)
@inline d_cn(m::CoupledKolluri, Δₙ_max::T) where T = one(T) - exp(-Δₙ_max / m.δₙ) * (1 + Δₙ_max / m.δₙ)
@inline d_o2(m::CoupledKolluri, cᵍᵇₘₐₓ) = m.dₒ₂ₘₐₓ * (one(cᵍᵇₘₐₓ) - exp(-cᵍᵇₘₐₓ / m.cᵍᵇ_char))

# cohesive stiffnesses
@inline function Kₙ(m::CoupledKolluri, cᵍᵇₘₐₓ::T1, Δₘₐₓ::Vec{dim_s, T2}, Δₙ) where {dim_s, T1, T2}
    (; Φₙ, δₙ) = m
    ℋ(Δₙ) = ℋᵣ(Δₙ, δₙ)
    # _T  = promote_type(T1, T2)
    Δₜₘₐₓ, Δₙₘₐₓ = split_jump(Δₘₐₓ)
    _Kₙ = (one(T1) - d_o2(m, cᵍᵇₘₐₓ)*ℋ(Δₙ)) * Φₙ/δₙ^2 * (one(T2) - d_n(m, Δₙₘₐₓ)*ℋ(Δₙ)) * (one(T2) - d_ct(m, Δₜₘₐₓ)*ℋ(Δₙ))
    return _Kₙ
end

@inline function Kₜ(m::CoupledKolluri, cᵍᵇₘₐₓ::T1, Δₘₐₓ::Vec{dim_s, T2}, Δₙ) where {dim_s, T1, T2}
    (; Φₜ, δₜ, δₙ) = m
    ℋ(Δₙ) = ℋᵣ(Δₙ, δₙ)
    # _T = promote_type(T1, T2)
    Δₜₘₐₓ, Δₙₘₐₓ = split_jump(Δₘₐₓ)
    _Kₜ = (one(T1) - d_o2(m, cᵍᵇₘₐₓ)*ℋ(Δₙ)) * Φₜ/δₜ^2 * (one(T2) - d_t(m, Δₜₘₐₓ)) * (one(T2) - d_cn(m, Δₙₘₐₓ))
    return _Kₜ
end

# cohesive tractions
@inline function T(m::CoupledKolluri, Δ::Vec{dim_s, T1}, Δₘₐₓ::Vec{dim_s, T2}, cᵍᵇ::T3, cᵍᵇₘₐₓ::T4) where {dim_s, T1, T2, T3, T4}
    (; Vₒ₂, cᵍᵇ₀) = m
    TT = promote_type(T1, T2, T3, T4)
    Δₜ, Δₙ = split_jump(Δ)
    _T = Vec{dim_s, TT}((Kₜ(m, cᵍᵇₘₐₓ, Δₘₐₓ, Δₙ) * Δₜ..., Kₙ(m, cᵍᵇₘₐₓ, Δₘₐₓ, Δₙ) * (Δₙ - Vₒ₂*(cᵍᵇ-cᵍᵇ₀))))
    return _T
end

@inline T̂ₙ(m::CoupledKolluri, Δ, Δₘₐₓ, cᵍᵇₘₐₓ) = Kₙ(m, cᵍᵇₘₐₓ, Δₘₐₓ, Δ[end]) * Δ[end]