@testset "CoupledKolluri material" begin
    D = 1e-10 # mm²/s 
    cᵍᵇ_char = 0.1e-15 # mol/mm²
    cᵍᵇ₀ = 0.0
    Vₒ₂ = 7.93 # mm³/mol
    δₙ = 1.5e-3 # mm
    δₜ = 1.5e-3 # mm

    σₘₐₓ = τₘₐₓ = 2e3

    Φₙ = σₘₐₓ * δₙ / (1. - (1. - exp(-1.0)) * Plastoxid.ℋᵣ(δₙ, δₙ))# N/mm
    Φₜ = τₘₐₓ * δₜ * exp(0.5) # N/mm
    M = 9.8e-3 # mm²/N 
    dₒ₂ₘₐₓ = 0.8
    R = 8.314e3 # N mm / (mol K)
    T = 700. + 273.15 # K

    # Parameter conversions
    @test Plastoxid.Φₙ(CoupledKolluri, σₘₐₓ, δₙ) ≈ Φₙ
    @test Plastoxid.Φₜ(CoupledKolluri, τₘₐₓ, δₜ) ≈ Φₜ
    @test Plastoxid.σₘₐₓ(CoupledKolluri, Φₙ, δₙ) ≈ σₘₐₓ
    @test Plastoxid.τₘₐₓ(CoupledKolluri, Φₜ, δₜ) ≈ τₘₐₓ


    material = Plastoxid.CoupledKolluri(D, cᵍᵇ_char, cᵍᵇ₀, Vₒ₂, δₙ, δₜ, Φₙ, Φₜ, M, dₒ₂ₘₐₓ, R, T)
    state = initial_material_state(material)

    # no loading
    Δ = zero(Vec{2})
    c = 0.0
    T, dTdΔ, dTdcᵍᵇ, state_temp = material_response(material, Δ, c, state)
    @test T == zero(Δ)
    @test dTdΔ[2,2] == Φₙ / δₙ^2 
    @test dTdΔ[1,1] == Φₜ / δₜ^2

    # tangential strength
    Δ = Vec((δₜ, 0.0))
    T, dTdΔ, dTdcᵍᵇ, state_temp = material_response(material, Δ, c, state)
    @test T[1] ≈ τₘₐₓ
    @test dTdΔ[1,1] ≈ 0.0

    # normal strength
    Δ = Vec((0.0, δₙ))
    T, dTdΔ, dTdcᵍᵇ, state_temp = material_response(material, Δ, c, state)
    @test T[2] ≈ σₘₐₓ

    # degeneration by oxygen
    Δ = zero(Vec{2})
    c = cᵍᵇ_char
    T, dTdΔ, dTdcᵍᵇ, state_temp = material_response(material, Δ, c, state)
    Δₒ₂ = Vₒ₂*(c-cᵍᵇ₀)
    @test (1.0 - dₒ₂ₘₐₓ*(1.0 - exp(-1))*Plastoxid.ℋᵣ(0.0, δₙ)) * Φₙ / δₙ^2 * (-Δₒ₂) ≈ T[2]
end