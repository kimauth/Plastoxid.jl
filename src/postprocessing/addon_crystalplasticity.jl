
function state_conjugates(
    ε::SymmetricTensor{2,3,T,6}, 
    material::CrystalViscoPlastic{S},
    state::CrystalViscoPlasticState{dim,T,M,S},
) where {T, S, dim, M}

    (; Eᵉ, H_iso, H_kin) = material

    εᵉ = inv(Eᵉ) ⊡ state.σ
    εᵖ = ε - εᵉ

    k = - state.κ / H_iso
    a = - state.α / H_kin

    return εᵖ, k, a
end

function strain_rates(
    material::CrystalViscoPlastic{S},
    σ::SymmetricTensor{2,3},
    α::SVector{S},
    λ::SVector{S},
) where S

    (; α_∞) = material

    τ = map(ms-> σ ⊡  ms, material.MS)
    dεᵖdt = zero(symmetric(first(material.MS)))
    for i=1:S
        dεᵖdt += λ[i] * sign(τ[i] - α[i]) * symmetric(material.MS[i])
    end
    # dεᵖdt = sum(λ .* (sign.(τ - α) .* symmetric.(material.MS)))
    dkdt = -λ
    dadt = - λ .* (sign.(τ - α) - α ./ α_∞)

    return dεᵖdt, dkdt, dadt
end

function dissipation(
    ::CrystalViscoPlastic{S},
    σ::SymmetricTensor{2,3},
    α::SVector{S},
    κ::SVector{S},
    dεᵖdt::SymmetricTensor{2,3},
    dkdt::SVector{S},
    dadt::SVector{S},
) where S

    𝔇 = σ ⊡ dεᵖdt + κ ⋅ dkdt + α ⋅ dadt

    return 𝔇
end