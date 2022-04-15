function test_surfacevectorvalues()
    ip_base = Lagrange{1, RefCube, 1}()
    ip = JumpInterpolation(ip_base)
    ip_geo = MidPlaneInterpolation(ip_base)
    qr = QuadratureRule{1,RefCube}(:lobatto, 2)
    cv = SurfaceVectorValues(qr, ip, ip_geo)

    X = [Vec((2.0, 0.0)), Vec((8.0, 4.0)), Vec((0.0, 3.0)),  Vec((6.0, 7.0))]
    reinit!(cv, X)

    # mid-point computation
    A = spatial_coordinate(cv, 1, X)
    B = spatial_coordinate(cv, 2, X)
    @test A ≈ Vec((1.0, 1.5))
    @test B ≈ Vec((7.0, 5.5))

    # 2D rotation matrix
    R = getR(cv, 1)
    @test R[:,1] ≈ 1/norm(B-A)*(B-A)
    @test R[1,2] ≈ -1/norm(B-A)*(B[2]-A[2])
    @test R[2,2] ≈ 1/norm(B-A)*(B[1]-A[1])


    # function value
    u = [2., 0., 3., 0.,
            4., 3., 4., 5., ]
    val_qp1 = function_value(cv, 1, u)
    @test val_qp1[1] ≈ 2.0
    @test val_qp1[2] ≈ 3.0
    val_qp2 = function_value(cv, 2, u)
    @test val_qp2[1] ≈ 1.0
    @test val_qp2[2] ≈ 5.0

    # function mid-plane value
    u_qp1 = function_midplane_value(cv, 1, u)
    @test u_qp1 ≈ Vec((3., 1.5))
    u_qp2 = function_midplane_value(cv, 2, u)
    @test u_qp2 ≈ Vec((3.5, 2.5))

    # function in-plane value
    Δ_A = function_value(cv, 1, u) ⋅ R # local jump
    Δ_B = function_value(cv, 2, u) ⋅ R # local jump
    Δ_mid_e = [Δ_A, Δ_B]
    @test function_inplane_value(cv, 1, Δ_mid_e) ≈ Δ_A
    @test function_inplane_value(cv, 2, Δ_mid_e) ≈ Δ_B

    # function gradient
    ∇Δ1 =  R' ⋅ function_gradient(cv, 1, u) ⋅ R
    ∇Δ2 =  R' ⋅ function_gradient(cv, 1, reinterpret(Vec{2,Float64}, u)) ⋅ R

    # function in-plane gradient
    ∇Δ3 = function_inplane_gradient(cv, 1, Δ_mid_e) ⋅ R
    ∇Δ4 = function_inplane_gradient(cv, 1, reinterpret(Float64, Δ_mid_e)) ⋅ R
    @test ∇Δ1 ≈ ∇Δ2 ≈ ∇Δ3 ≈ ∇Δ4 # different ways to compute the same thing

    # check that local gradients are correct
    @test ∇Δ1[:,1] ≈ (Δ_B - Δ_A) / norm(B-A) # local in-plane gradient
    @test all(∇Δ1[:,2] .< 1e-16) # local gradient normal to plane must be 0

    # test integration
    l = 0.0
    for qp in 1:getnquadpoints(cv)
        l += getdetJdA(cv, qp)
    end
    @test l ≈ norm(B-A)

    # local coordinates
    x = [X[i] ⋅ R for i=1:length(X)]
    @test x[1][2] == x[2][2]
    @test x[3][2] == x[4][2]
    @test x[1][1] == x[3][1]
    @test x[2][1] == x[4][1]

    qp = 1
    dXdξ = [0.0, 0.0]
    dxdξ = [0.0, 0.0]
    for i in 1:length(X)
        dXdξ += cv.dMdξ[i, qp].data[1] * X[i]
        dxdξ += cv.dMdξ[i, qp].data[1] * x[i]
    end

    # normal computation
    N̂ = 1/norm(dXdξ)*[-dXdξ[2], dXdξ[1]]
    n̂ = [0., 1.]
    @test Tensor{1,2}(n̂) ⋅ R' ≈ N̂
    @test N̂ ≈ R[:,end]

    # rotate jacobian around (example)
    @test Tensor{2,2}(hcat(dXdξ, N̂))' ⋅ R ≈ hcat(dxdξ, n̂)
    @test R ⋅ Tensor{2,2}(hcat(dxdξ, n̂)) ≈ hcat(dXdξ, N̂)

    @test inv(Tensor{2,2}(hcat(dXdξ, N̂))) ≈ inv(Tensor{2,2}(hcat(dxdξ, n̂))) ⋅ R'
end

function test_surfacescalarvalues()
    ip_base = Lagrange{1, RefCube, 1}()
    ip = JumpInterpolation(ip_base)
    ip_geo = MidPlaneInterpolation(ip_base)
    qr = QuadratureRule{1,RefCube}(:lobatto, 2)
    cv = SurfaceScalarValues(qr, ip, ip_geo)

    X = [Vec((2.0, 0.0)), Vec((8.0, 4.0)), Vec((0.0, 3.0)),  Vec((6.0, 7.0))]
    reinit!(cv, X)

    # mid-point computation
    A = spatial_coordinate(cv, 1, X)
    B = spatial_coordinate(cv, 2, X)
    @test A ≈ Vec((1.0, 1.5))
    @test B ≈ Vec((7.0, 5.5))

    # 2D rotation matrix
    R = getR(cv, 1)#Plastoxid.getR(cv, 1)
    @test R[:,1] ≈ 1/norm(B-A)*(B-A)
    @test R[1,2] ≈ -1/norm(B-A)*(B[2]-A[2])
    @test R[2,2] ≈ 1/norm(B-A)*(B[1]-A[1])


    # function value
    c = [2., 3., 4., 4. ]
    @test function_value(cv, 1, c) ≈ 2.0
    @test function_value(cv, 2, c) ≈ 1.0

    # function in-plane value
    Δc_A = function_value(cv, 1, c) 
    Δc_B = function_value(cv, 2, c)
    Δc_mid_e = [Δc_A, Δc_B]
    @test function_inplane_value(cv, 1, Δc_mid_e) ≈ Δc_A
    @test function_inplane_value(cv, 2, Δc_mid_e) ≈ Δc_B

    # function mid-plane value
    @test function_midplane_value(cv, 1, c) ≈ 3.0
    @test function_midplane_value(cv, 2, c) ≈ 3.5

    # function gradient
    ∇Δc1 =  function_gradient(cv, 1, c) ⋅ R

    # function in-plane gradients
    ∇Δc2 = function_inplane_gradient(cv, 1, Δc_mid_e) ⋅ R
    @test ∇Δc1 ≈ ∇Δc2

    # check that local gradients are correct
    @test ∇Δc1[1] ≈ (Δc_B - Δc_A) / norm(B-A) # local in-plane gradient
    @test ∇Δc1[2] < 1e-16 # local gradient normal to plane must be 0

    # function mid-plane gradient
    cᵐⁱᵈ_A = function_midplane_value(cv, 1, c)
    cᵐⁱᵈ_B = function_midplane_value(cv, 2, c)
    ∇cᵐⁱᵈ = function_midplane_gradient(cv, 1, c) ⋅ R
    @test ∇cᵐⁱᵈ[1] ≈ (cᵐⁱᵈ_B - cᵐⁱᵈ_A) / norm(B-A)
    @test ∇cᵐⁱᵈ[2] ≈ 0.0

    # test integration
    l = 0.0
    for qp in 1:getnquadpoints(cv)
        l += getdetJdA(cv, qp)
    end
    @test l ≈ norm(B-A)

    # local coordinates
    x = [X[i] ⋅ R for i=1:length(X)]
    @test x[1][2] == x[2][2]
    @test x[3][2] == x[4][2]
    @test x[1][1] == x[3][1]
    @test x[2][1] == x[4][1]

    qp = 1
    dXdξ = [0.0, 0.0]
    dxdξ = [0.0, 0.0]
    for i in 1:length(X)
        dXdξ += cv.dMdξ[i, qp].data[1] * X[i]
        dxdξ += cv.dMdξ[i, qp].data[1] * x[i]
    end

    # normal computation
    N̂ = 1/norm(dXdξ)*[-dXdξ[2], dXdξ[1]]
    n̂ = [0., 1.]
    @test Tensor{1,2}(n̂) ⋅ R' ≈ N̂
    @test N̂ ≈ R[:,end]

    # rotate jacobian around (example)
    @test Tensor{2,2}(hcat(dXdξ, N̂))' ⋅ R ≈ hcat(dxdξ, n̂)
    @test R ⋅ Tensor{2,2}(hcat(dxdξ, n̂)) ≈ hcat(dXdξ, N̂)

    @test inv(Tensor{2,2}(hcat(dXdξ, N̂))) ≈ inv(Tensor{2,2}(hcat(dxdξ, n̂))) ⋅ R'
end

@testset "SurfaceValues" begin
    test_surfacevectorvalues()
    test_surfacescalarvalues()
end