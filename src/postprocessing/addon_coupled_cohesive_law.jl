
function damage_variables(material::CoupledKolluri, state::CoupledKolluriState) 
    (; Δₘₐₓ, cᵍᵇₘₐₓ) = state
    Δₜ_max, Δₙ_max = split_jump(Δₘₐₓ)

    dn = d_n(material, Δₙ_max)
    dct = d_ct(material, Δₜ_max)
    dt = d_t(material, Δₜ_max)
    dcn = d_cn(material, Δₙ_max)
    do2 = d_o2(material, cᵍᵇₘₐₓ)

    return dn, dct, dt, dcn, do2
end

function damage_time_derivatives(
    ::CoupledKolluri,
    dn, dct, dt, dcn, do2,
    dn_old, dct_old, dt_old, dcn_old, do2_old,
    Δt::Float64,
)
    ∂dn∂t = (dn - dn_old) / Δt
    ∂dct∂t = (dct - dct_old) / Δt
    ∂dt∂t = (dt - dt_old) / Δt
    ∂dcn∂t = (dcn - dcn_old) / Δt
    ∂do2∂t = (do2 - do2_old) / Δt

    return ∂dn∂t, ∂dct∂t, ∂dt∂t, ∂dcn∂t, ∂do2∂t
end

function damage_driving_forces(
    m::CoupledKolluri,
    Δ::Vec{dim, T},
    cᵍᵇ::T,
    dn,
    dct,
    dt,
    dcn,
    do2,
) where {dim, T}

    (; Φₜ, Φₙ, δₜ, δₙ, Vₒ₂, cᵍᵇ₀) = m
    Δₜ, Δₙ = split_jump(Δ)
    ℋ(Δₙ) = ℋᵣ(Δₙ, δₙ)

    ∂Ψ∂dn = -ℋ(Δₙ) * (one(T) - do2*ℋ(Δₙ)) * Φₙ/δₙ^2 * (one(T) - dct*ℋ(Δₙ)) * (Δₙ/2 - Vₒ₂*(cᵍᵇ - cᵍᵇ₀)) * Δₙ
    ∂Ψ∂dct = -ℋ(Δₙ) * (one(T) - do2*ℋ(Δₙ)) * Φₙ/δₙ^2 * (one(T) - dn*ℋ(Δₙ)) * (Δₙ/2 - Vₒ₂*(cᵍᵇ - cᵍᵇ₀)) * Δₙ
    ∂Ψ∂dt = -(one(T) - do2*ℋ(Δₙ)) * 1/2 * Φₜ/δₜ^2 * (one(T) - dcn) * Δₜ ⋅ Δₜ
    ∂Ψ∂dcn = -(one(T) - do2*ℋ(Δₙ)) * 1/2 * Φₜ/δₜ^2 * (one(T) - dt) * Δₜ ⋅ Δₜ

    _Kₙ = Φₙ/δₙ^2 * (one(T) - dn*ℋ(Δₙ)) * (one(T) - dct*ℋ(Δₙ))
    _Kₜ = Φₜ/δₜ^2 * (one(T) - dt) * (one(T) - dcn)
    ∂Ψ∂do2 = -ℋ(Δₙ) * (_Kₙ * (Δₙ/2 - Vₒ₂*(cᵍᵇ - cᵍᵇ₀)) * Δₙ + 1/2 * _Kₜ * Δₜ ⋅ Δₜ)

    return ∂Ψ∂dn, ∂Ψ∂dct, ∂Ψ∂dt, ∂Ψ∂dcn, ∂Ψ∂do2
end

function dissipation(
    ::CoupledKolluri,
    ∂Ψ∂dn, ∂Ψ∂dct, ∂Ψ∂dt, ∂Ψ∂dcn, ∂Ψ∂do2,
    ∂dn∂t, ∂dct∂t, ∂dt∂t, ∂dcn∂t, ∂do2∂t
)
    𝔇 = -(∂Ψ∂dn*∂dn∂t + ∂Ψ∂dct*∂dct∂t + ∂Ψ∂dt*∂dt∂t + ∂Ψ∂dcn*∂dcn∂t + ∂Ψ∂do2*∂do2∂t)

    return 𝔇
end

###########################################################
# convenience wrappers
###########################################################

function damage_time_derivatives(
    material::CoupledKolluri,
    state::CoupledKolluriState,
    old_state::CoupledKolluriState,
    Δt::Float64,
)
    dn, dct, dt, dcn, do2 = damage_variables(material, state)
    dn_old, dct_old, dt_old, dcn_old, do2_old = damage_variables(material, old_state)
    return damage_time_derivatives(material, dn, dct, dt, dcn, do2, dn_old, dct_old, dt_old, dcn_old, do2_old, Δt)
end

function damage_driving_forces(
    m::CoupledKolluri,
    Δ::Vec{dim, T},
    cᵍᵇ::T,
    state::CoupledKolluriState{dim,T},
) where {dim, T}

    dn, dct, dt, dcn, do2 = damage_variables(material, state)
    return damage_driving_forces(m, Δ, cᵍᵇ, dn, dct, dt, dcn, do2)
end

function dissipation(
    m::CoupledKolluri,
    Δ::Vec{dim, T},
    cᵍᵇ::T,
    state::CoupledKolluriState{dim,T},
    old_state::CoupledKolluriState{dim,T},
) where {dim, T}

    dn, dct, dt, dcn, do2 = damage_variables(material, state)
    dn_old, dct_old, dt_old, dcn_old, do2_old = damage_variables(material, old_state)
    ∂dn∂t, ∂dct∂t, ∂dt∂t, ∂dcn∂t, ∂do2∂t = damage_time_derivatives(material, dn, dct, dt, dcn, do2, dn_old, dct_old, dt_old, dcn_old, do2_old, Δt)
    ∂Ψ∂dn, ∂Ψ∂dct, ∂Ψ∂dt, ∂Ψ∂dcn, ∂Ψ∂do2 = damage_driving_forces(m, Δ, cᵍᵇ, dn, dct, dt, dcn, do2)

    return dissipation(m, ∂Ψ∂dn, ∂Ψ∂dct, ∂Ψ∂dt, ∂Ψ∂dcn, ∂Ψ∂do2, ∂dn∂t, ∂dct∂t, ∂dt∂t, ∂dcn∂t, ∂do2∂t)
end