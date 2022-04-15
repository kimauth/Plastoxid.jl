
@testset "CellBuffer" begin
    ## MechanicalEquilibrium
    @testset "CellBuffer MechanicalEquilibrium" begin
        materials = [LinearElastic(E=200e3, ν=0.3)]
        dim_type = PlaneStrain()
        t = 2.0

        problem = MechanicalEquilibrium(materials, dim_type, t)

        ip = Lagrange{2, RefCube, 2}()
        ip_geo = Lagrange{2,RefCube,1}()
        qr = QuadratureRule{2,RefCube}(2)
        qr_face = QuadratureRule{1,RefCube}(2)

        cb = cellbuffer(problem, ip, ip_geo, qr, qr_face)
        typeof(cb.cv)
        @test isa(cb.cv, CellVectorValues)
        @test isa(cb.fv, FaceVectorValues)
    end

    ## Diffusion
    @testset "CellBuffer Diffusion" begin
        materials = [DiffusionCoefficient(1.0)]
        dim_type = MaterialModels.Dim{1}()
        A = 4.0

        problem = FicksLaw(materials, dim_type, A)

        ip = Lagrange{1, RefCube, 2}()
        ip_geo = Lagrange{1,RefCube,1}()
        qr = QuadratureRule{1,RefCube}(2)
        qr_face = QuadratureRule{0,RefCube}(2)

        cb = cellbuffer(problem, ip, ip_geo, qr, qr_face)
        @test isa(cb.cv, CellScalarValues)
        @test isa(cb.fv, FaceScalarValues)
    end

    ## Cohesive Zone 
    @testset "CellBuffer CohesiveZone" begin
        σₘₐₓ = 400.
        τₘₐₓ=200.
        δₙ=0.002
        δₜ=0.002
        Φₙ = xu_needleman_Φₙ(σₘₐₓ, δₙ)
        Φₜ = xu_needleman_Φₜ(τₘₐₓ, δₜ)
        materials = [XuNeedleman(;σₘₐₓ, τₘₐₓ, Φₙ, Φₜ, Δₙˢ=0.01)]

        dim_type = MaterialModels.Dim{2}()
        t = 2.0

        problem = Interface(materials, dim_type, t)

        ip_base_f = Lagrange{1, RefCube, 1}()
        ip = JumpInterpolation(ip_base_f)
        ip_base_geo = Lagrange{1, RefCube, 1}()
        ip_geo = MidPlaneInterpolation(ip_base_geo)
        qr = QuadratureRule{1,RefCube}(:lobatto, 3)

        cv, fv = Plastoxid.get_values(problem, ip, ip_geo, qr, nothing)

        cb = cellbuffer(problem, ip, ip_geo, qr, nothing)
        
        @test isa(cb.cv, SurfaceVectorValues)
        @test isa(cb.fv, Nothing)
    end

    @testset "CoupledCellBuffer InterfaceFicksLaw" begin
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
        material = Plastoxid.CoupledKolluri(D, cᵍᵇ_char, cᵍᵇ₀, Vₒ₂, δₙ, δₜ, Φₙ, Φₜ, M, dₒ₂ₘₐₓ, R, T)

        materials = [material]
        dim_type = MaterialModels.Dim{2}()
        t = 2.0
        shell_thickness = 1e-4

        ip_base_f = Lagrange{1, RefCube, 1}()
        ip = JumpInterpolation(ip_base_f)
        ip_base_geo = Lagrange{1, RefCube, 1}()
        ip_geo = MidPlaneInterpolation(ip_base_geo)
        qr = QuadratureRule{1,RefCube}(:lobatto, 3)
        qr_face = QuadratureRule{0,RefCube}(1)

        # manual set-up here, matches interpolations above
        dof_ranges = Dict(:u => 1:8, :c => 9:12)

        problem = InterfaceFicksLaw(materials, shell_thickness, dim_type, t, dof_ranges)

        cb = cellbuffer(problem, ip, ip, ip_geo, qr, qr_face)

        @test isa(cb.cv_u, SurfaceVectorValues)
        @test isa(cb.cv_c, SurfaceScalarValues)
        @test isa(cb.fv_c, SurfaceFaceScalarValues)
    end
end