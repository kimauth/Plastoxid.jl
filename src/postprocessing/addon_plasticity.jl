################################################################
# Plastic
################################################################

function state_conjugates(
    ε::SymmetricTensor{2,3,T,6}, 
    material::Plastic,
    state::PlasticState{3},
) where T
    (; Eᵉ, r, H) = material

    σ = Eᵉ ⊡ (ε - state.εᵖ)
    k = - state.κ / (r * H)
    a = - 3/2 / ((1-r)*H) * state.α

    return σ, k, a
end

function strain_rates(
    material::Plastic,
    σ::SymmetricTensor{2,3},
    κ::Real,
    α::SymmetricTensor{2,3},
    λ::Real,
)
    (; κ_∞, α_∞) = material

    # avoid recomputations
    σʳᵉᵈ = σ - α
    σʳᵉᵈ_dev = dev(σʳᵉᵈ)
    σₑʳᵉᵈ = sqrt(3/2) * norm(σʳᵉᵈ_dev)
    α_dev = dev(α)

    # flow + hardening rules
    ν = 3/2 * σʳᵉᵈ_dev / σₑʳᵉᵈ
    ζ_κ = -1. + κ / κ_∞
    ζ_α = -ν + 3. / (2α_∞) * α_dev

    # evolution equations
    dεᵖdt = λ * ν
    dkdt = λ * ζ_κ
    dadt = λ * ζ_α

    return dεᵖdt, dkdt, dadt
end

function dissipation(
    ::Plastic, # could perhaps be more general
    σ::SymmetricTensor{2,3},
    κ::Real,
    α::SymmetricTensor{2,3},
    dεᵖdt::SymmetricTensor{2,3},
    dkdt::Real,
    dadt::SymmetricTensor{2,3},
)
    # dissipation
    𝔇 = σ ⊡ dεᵖdt + κ * dkdt + α ⊡ dadt

    return 𝔇
end

## convenience wrappers if intermediate results are not needed
function strain_rates(
    ε::SymmetricTensor{2,3}, 
    material::Plastic,
    state::PlasticState{3},
    Δt::Float64,
)
    σ, _, _ = state_conjugates(ε, material, state)
    (; κ, α, μ) = state

    λ = μ / Δt
    return strain_rates(material, σ, κ, α, λ)
end

function dissipation(
    ε::SymmetricTensor{2,3}, 
    material::Plastic,
    state::PlasticState{3},
    Δt::Float64,
)
    σ, _, _ = state_conjugates(ε, material, state)
    (; κ, α, μ) = state

    λ = μ / Δt
    dεᵖdt, dkdt, dadt = strain_rates(material, σ, κ, α, λ)

    return dissipation(material, σ, κ, α, dεᵖdt, dkdt, dadt)
end