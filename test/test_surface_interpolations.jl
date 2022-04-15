@testset "SurfaceInterpolation" begin
    ip_base = Lagrange{1, RefCube, 1}()
    ip = JumpInterpolation(ip_base)
    ip_geo = MidPlaneInterpolation(ip_base)
    qr = QuadratureRule{1,RefCube}(:lobatto, 2)

    dNdξ, N = gradient(ξ -> Ferrite.value(ip, 1, ξ), qr.points[1], :all)
    @test N ≈ -1.0
    @test dNdξ[1] == 0.5
    dNdξ, N = gradient(ξ -> Ferrite.value(ip, 2, ξ), qr.points[1], :all)
    @test N < 1e-14
    @test dNdξ[1] == -0.5
    dNdξ, N = gradient(ξ -> Ferrite.value(ip, 3, ξ), qr.points[1], :all)
    @test N ≈ 1.0
    @test dNdξ[1] == -0.5
    dNdξ, N = gradient(ξ -> Ferrite.value(ip, 4, ξ), qr.points[1], :all)
    @test N < 1e-14
    @test dNdξ[1] == 0.5
end