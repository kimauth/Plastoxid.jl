struct DiffusionCoefficient <: AbstractMaterial
    D::Float64
end

MaterialModels.get_cache(::DiffusionCoefficient) = nothing