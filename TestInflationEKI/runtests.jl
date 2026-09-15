using Test
using LinearAlgebra
using Random
using Statistics

include(joinpath(@__DIR__, "..", "Inversion", "InflationEKI.jl"))
include(joinpath(@__DIR__, "..", "Inversion", "CMAES.jl"))
include(joinpath(@__DIR__, "NonlinearFunctions.jl"))

function sample_covariance(points)
    anomalies = (points .- mean(points, dims=2)) ./ sqrt(size(points, 2) - 1)
    return anomalies * anomalies'
end

@testset "nonlinear benchmark definitions" begin
    for name in ("rastrigin", "rotated_rastrigin", "monotone_cubic")
        args = make_test_args(name; dim=8, seed=31)
        @test func_phi(args.ref_theta, args) ≈ 0.0 atol=1e-12
        @test length(func_F(args.ref_theta, args)) >= args.dim
    end

    args = make_test_args("paired_rosenbrock"; dim=8, seed=31)
    @test func_phi(args.ref_theta, args) ≈ 0.0 atol=1e-12
    @test length(func_F(args.ref_theta, args)) == 8
    standard_start = repeat([-1.2, 1.0], 4)
    @test func_phi(standard_start, args) ≈ 4 * 24.2

    rastrigin = make_test_args("rastrigin"; dim=3)
    point = fill(0.5, 3)
    @test func_phi(point, rastrigin) ≈ sum(point.^2 .+ 20 .* sin.(pi .* point).^2)
end

@testset "projected corrections preserve EAKI covariance" begin
    rng = MersenneTwister(4)
    d, J = 7, 5
    G = randn(rng, d, d)
    forward(theta) = G * theta
    theta0 = randn(rng, d, J) .+ 0.7
    y = randn(rng, d)
    sigma_y = Matrix{Float64}(I, d, d)

    base = EKI_Run(forward, copy(theta0), sigma_y, y;
        filter_type="EAKI", Δτ=0.4, N_iter=1)

    for mode in ("sequential", "joint")
        Random.seed!(9)
        projected = EKI_Run(forward, copy(theta0), sigma_y, y;
            filter_type="dropout-EAKI", Δτ=0.4, N_iter=1,
            dropout_correction_mode=mode)
        @test sample_covariance(projected.θ[end]) ≈
              sample_covariance(base.θ[end]) rtol=1e-9 atol=1e-10
    end
end

@testset "joint is the direct raw update" begin
    args = make_test_args("rastrigin"; dim=6, seed=7)
    forward(theta) = func_F(theta, args)
    rng = MersenneTwister(13)
    d, J = 6, 8
    theta0 = 1.0 .+ 2.0 .* randn(rng, d, J)
    y = zeros(length(forward(zeros(6))))
    sigma_y = Matrix{Float64}(I, length(y), length(y))
    evaluations = Ref(0)
    counted_forward(theta) = begin
        evaluations[] += ndims(theta) == 1 ? 1 : size(theta, 2)
        forward(theta)
    end

    Random.seed!(18)
    result = EKI_Run(counted_forward, theta0, sigma_y, y;
        filter_type="dropout-EAKI", Δτ=0.5, N_iter=2,
        dropout_correction_mode="joint")

    @test all(isfinite, result.θ[end])
    @test length(result.θ) == 3
    @test evaluations[] == J + 2 * (2 * J + 1)
end

@testset "original PyCall CMA-ES wrapper" begin
    objective(x) = sum(abs2, x)
    result = run_cmaes(objective, fill(3.0, 5);
        sigma0=1.0, max_iter=60, popsize=10, seed=1)
    @test result.best_f < 1e-4
    @test all(diff(result.best_f_history) .<= 0)
end
