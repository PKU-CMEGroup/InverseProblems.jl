using LinearAlgebra
using Random


"""
    func_F(theta, args)

Nonlinear least-squares benchmark forward maps.

Supported:
- "rastrigin"
- "rosenbrock"
- "weakly_nonlinear"

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
        A=2.0,
        epsilon=1.0,
        condition_number=10.0)

    rng=MersenneTwister(seed)


    if name=="rastrigin"

        ref_theta=zeros(dim)

        return (
            name=name,
            dim=dim,
            ref_theta=ref_theta,
            A=A
        )


    elseif name=="rosenbrock"

        ref_theta=ones(dim)

        return (
            name=name,
            dim=dim,
            ref_theta=ref_theta
        )


    elseif name=="weakly_nonlinear"

        dim_y = 20

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


    else

        error("Unknown nonlinear function")

    end

end