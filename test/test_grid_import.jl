input_path = joinpath(@__DIR__, "coh_tris_2x2.inp")
mesh = abaqus2Ferrite_grid(input_path)
@test getnnodes(mesh) == 12
@test getncells(mesh) == 10

# test expemplary cells for cell type and correct node numbers
@test isa(mesh.cells[5], CohesiveCell)
@test mesh.cells[5].nodes == (5,11,1,9)

@test isa(mesh.cells[1], Triangle)
@test mesh.cells[1].nodes == (1,9,10)

# test for correct cellsets
@test getcellset(mesh, "COH2D4") == Set([5,6])
@test getcellset(mesh, "CPS3") == Set(setdiff(1:10, 5:6))
