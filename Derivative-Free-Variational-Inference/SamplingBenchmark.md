# Sampling Benchmark Overview

**[Benchmark.jl](Benchmark.jl)** evaluates the performance of 3 different sampling methods on 5 different test distributions. The comparison focuses on accuracy, computational efficiency, and convergence behavior.

## Test Distributions

The benchmark tests five posterior distributions, see [Multimodal.jl](Multimodal.jl) for details:

1. **Gaussian**: A standard Gaussian distribution with correlated components
2. **Four Modes**: A multi-modal distribution with four distinct modes
3. **Circle**: A distribution concentrated on a circle in 2D space
4. **Banana**: A curved, banana-shaped distribution (also known as Rosenbrock distribution)
5. **Funnel**: A funnel-shaped distribution with varying scales across dimensions

## Sampling Methods

Three sampling algorithms are compared:

1. **GMBBVI (Gaussian Mixture Black Box Variational Inference)**: An derivative-free Gaussian mixture variational inference method with adaptive step tuning. See *<https://arxiv.org/abs/2601.14855>* for details.
2. **MCMC using Stretch Move**: An affine-invariant and derivative-free ensemble sampler with stretch move proposal.
   Implemented using the [`emcee`](https://emcee.readthedocs.io/en/stable/) Python package.
3. **WALNUTS**: A No-U-Turn Sampler (NUTS) variant with adaptive integrator *<https://github.com/bob-carpenter/walnuts>* , which uses adaptive leapfrog integrator with dual time stepping and includes warmup phase for parameter tuning.

## Benchmark Parameters

### General Settings

- **Random Seed**: 111 (for reproducibility)
- **Problem Dimension**: $N\_x = 50$

### GMBBVI Parameters

- **Number of Iterations**: 500
- **Maximum Time Step**: dt = 0.9
- **Number of Samples per Component**: $ 4 N\_x = 200$
- **Initial Weights**: Uniform, **Initial Means**: Random samples from $\mathcal{N}(0, I)$, **Initial Covariances**: Identity matrix

### Stretch-Move MCMC Parameters

- **Walkers**: 500 walkers, initialization from $\mathcal{N}(0, I)$
- **Number of Iterations**: 20,000
- **Thinning**: 20 (save every 20th sample)

### WALNUTS Parameters

- **Total Samples**: 5,000
- **Number of Steps**: M = 12
- **Initial Step Size**: H0 = 0.3
- **Initial Delta**: delta0 = 0.3
- **Warmup Iterations**: 1,000
- **Adaptation Target Delta**: 0.6

## Evaluation Metrics

The benchmark evaluates:

1. **Total Variation (TV) Error**: L1-norm difference between estimated and true posterior densities
2. **Computational Time**: Wall-clock time for each method
3. **Sample Statistics**: Mean and covariance estimates compared to true values

## Result and Analysis

All experiments were  configured to use exactly one compute node and four CPU cores (i.e., --nodes=1 --ntasks=4). No GPU acceleration was used.

- Result: the benchmark generates: `Benchmark_50_Dim.pdf` containing all visualizations and sample statistics, and timing information for each method and distribution:

| Run time (s) | GMBBVI | MCMC    | WALNUTS |
| ------------ | ------ | ------- | ------- |
| Gaussian     | 181.14 | 189.59  | 93.99   |
| Four\_modes  | 140.42 | 201.69  | 95.25   |
| Circle       | 248.53 | 194.50  | 91.42   |
| Banana       | 136.37 |  193.44 | 246.99  |
| Funnel       | 176.63 | 199.40  | 169.84  |

- Analysis:

1. We mention that GMBBVI calls the potential function $\Phi\_R$ less than MCMC and WALNUTS: on each example, **GMBBVI calls $\Phi\_R$ for $4\time 10^6$ times, while Stretch-Move MCMC call it for $10^7$ times( It's difficult to determine how many times WALNUTS calls $\Phi\_R$ due to its interface limitations).** However, the computational cost of $\log \rho\_a^{\rm GM}$ leads to the higher cost.
2. GMBBVI approximates the true posterior distribution better than MCMC and WALNUTS in all example distribution except the Funnel distribution. The annealing process of GMBBVI helps to explore further modes of the posterior distribution.
3. On the Funnel distribution, GMBBVI approximates the true posterior distribution weakly than MCMC and WALNUTS, and the modes lies on the line ${x\_2=x\_3=\cdots=x\_{50}=0}$. We claim that the Funnel distribution is a pathological case for GMBBVI, which means that Gaussian mixture family cannot approximate the true posterior distribution well in this case.

## Usage

Make sure your julia ENV\["PYTHON"] points to the correct python executable with the required packages (**emcee**, scipy, numpy, matplotlib) installed. Run the benchmark script with:

```julia
julia Benchmark.jl
```

The script will automatically:

1. Set up Python paths for required packages
2. Run all three methods on each test distribution
3. Generate comparative visualizations
4. Save results to PDF file

