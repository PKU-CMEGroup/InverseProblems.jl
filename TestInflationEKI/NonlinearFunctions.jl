using LinearAlgebra
using Random


"""
    func_F(theta, args)

Nonlinear least-squares benchmark forward maps.

Supported:
- "rastrigin"
- "rotated_rastrigin"
- "paired_rosenbrock"
- "rosenbrock"
- "weakly_nonlinear"
- "monotone_cubic"

Objective:

    func_phi(theta,args)=0.5*||func_F(theta,args)||^2
"""
function func_F(theta, args)

    name = args.name
    dim = args.dim

    length(theta)==dim ||
        throw(DimensionMismatch("Wrong dimension"))

    if name == "rastrigin"

        A = args.A

        r1 = theta
        r2 = sqrt(2A) .* sin.(pi .* theta)

        return sqrt(2) .* vcat(r1,r2)

    elseif name == "rotated_rastrigin"

        z = args.B * (theta - args.ref_theta)
        r1 = z
        r2 = sqrt(2 * args.A) .* sin.(pi .* z)

        return sqrt(2) .* vcat(r1, r2)


    elseif name == "paired_rosenbrock"

        iseven(dim) || error("paired_rosenbrock requires even dimension")
        residual = zeros(eltype(theta), dim)

        for i in 1:div(dim, 2)
            x = theta[2i-1]
            y = theta[2i]
            residual[2i-1] = sqrt(200) * (y - x^2)
            residual[2i] = sqrt(2) * (1 - x)
        end

        return residual


    elseif name == "rosenbrock"

        d = dim
        @assert d%2==0 "Rosenbrock function requires even dimension"
        residual = zeros(d)

        for i in 1:div(d,2)
            residual[2i-1] =
                10*(theta[2i]-theta[2i-1]^2)

            residual[2i] =
                1-theta[2i-1]
        end

        return residual


    elseif name == "weakly_nonlinear"

        z = args.G * theta

        return z + args.epsilon * tanh.(z) - args.y


    elseif name == "monotone_cubic"

        z = args.B * theta
        z_ref = args.B * args.ref_theta
        ψ(v) = v .+ args.alpha .* v.^3

        return ψ(z) - ψ(z_ref)


    else

        error("Unknown nonlinear function")

    end

end



function func_phi(theta,args)

    F = func_F(theta,args)

    return 0.5*sum(abs2,F)

end




function make_test_args(
        name;
        dim=100,
        seed=1234,
        A=10.0,
        epsilon=1.0,
        condition_number=10.0,
        alpha=2.0)

    rng=MersenneTwister(seed)


    if name=="rastrigin"

        ref_theta=zeros(dim)

        return (
            name=name,
            dim=dim,
            ref_theta=ref_theta,
            A=A
        )


    elseif name=="rotated_rastrigin"

        ref_theta=zeros(dim)
        B=Matrix(qr(randn(rng,dim,dim)).Q)

        return (
            name=name,
            dim=dim,
            ref_theta=ref_theta,
            A=A,
            B=B
        )


    elseif name=="paired_rosenbrock"

        iseven(dim) || error("paired_rosenbrock requires even dimension")
        ref_theta=ones(dim)

        return (
            name=name,
            dim=dim,
            ref_theta=ref_theta
        )


    elseif name=="rosenbrock"

        ref_theta=ones(dim)

        return (
            name=name,
            dim=dim,
            ref_theta=ref_theta
        )


    elseif name=="weakly_nonlinear"

        dim_y = min(20, dim)

        # prescribed spectrum
        U=Matrix(qr(randn(rng,dim,dim)).Q)[1:dim_y,:]
        V=Matrix(qr(randn(rng,dim,dim)).Q)


        sigma =
            exp.(
                range(
                    -log(condition_number)/2,
                    log(condition_number)/2,
                    length=dim
                )
            )


        G =
            U*Diagonal(sigma)*V'


        ref_theta = randn(rng,dim)
        ref_z = G * ref_theta

        y =
            ref_z + epsilon * tanh.(ref_z)


        return (
            name=name,
            dim=dim,
            ref_theta=ref_theta,
            G=G,
            epsilon=epsilon,
            y=y,
            condition_number=condition_number
        )


    elseif name=="monotone_cubic"

        B=Matrix(qr(randn(rng,dim,dim)).Q)
        ref_theta=randn(rng,dim)

        return (
            name=name,
            dim=dim,
            ref_theta=ref_theta,
            B=B,
            alpha=alpha
        )


    else

        error("Unknown nonlinear function")

    end

end
