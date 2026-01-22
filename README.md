# InverseProblems

<img src="Figs/InverseProblems.png" width="800" />

InverseProblems is a

* educational **Inverse Problem** or **Numerical Method** library. 
The goal is to provide students with a light-weighted code to explore these areas 
and interactive lectures with amazing [Jupyter Notebook](https://jupyter.org/).
* benchmark repository originally designed to test unscented Kalman inversion and other **derivative-free inverse methods**. 
The goal is to provide reseachers with access to various inverse problems, 
while enabling researchers to quickly and easily develop and test novel inverse methods.

## Code Struction
* All the inverse methods are in *Inversion* folder
* Each other folder contains one category of inverse problems

## Tutorial
Let's start! (⚠️ under construction)



* Overview
    * What are inverse problems, why are they important?
    * [Bayesian inversion, Bayesian inference, and Bayesian calibration](Lectures/Bayesian.ipynb) 
    * [Probability density function space](Lectures/PDFSpace.ipynb) 
* [Probabilistic approaches](Lectures/Probabilistic.ipynb) 
    * Invariant and ergodic measures
      * [Langevin dynamics](Lectures/Langevin.ipynb) 
      * [Markov Chain Monte Carlo methods](Lectures/MonteCarlo.ipynb) 
      * Interacting particle methods
    * [Variational inference](Lectures/VariationalInference.ipynb)
      * [Gaussian variational inference](Lectures/GaussianVariationalInference.ipynb)
      * [Stein variational inference](Lectures/SteinVariationalInference.ipynb)
      * [Wasserstein gradient flow](Lectures/WassersteinGradientFlow.ipynb)
      * [Affine invariant gradient flows](Lectures/AffineInvariance.ipynb)
      * [Derivative Free Gaussian Mixture Variational Inference](Lectures/DerivativeFreeVariationalInference.ipynb)
      * [Gaussian Mixture Black-Box Variational Inference](Lectures/BlackBoxVariationalInference.ipynb)
    * Coupling ideas
      * [Filtering](Lectures/Filtering.ipynb)
         * [Kalman filters](Lectures/KalmanFilters.ipynb)
      * [Inversion](Lectures/Inversion.ipynb)
         * [Sequential Monte Carlo method](Lectures/MonteCarlo.ipynb)
         * [Kalman inversion part I : stochastic dynamical system](Lectures/KalmanInversionPartI.ipynb)
         * [Kalman inversion part II : implementation](Lectures/KalmanInversionPartII.ipynb)
      * Transport map
    * [When is posterior distribution close to Gaussian](Lectures/Posterior.ipynb)
    * All models are wrong
* Examples
   * Linear inverse problems
      * [Well-determined, under-determined, and over-determined inverse problems](Linear/Linear-2-parameter.ipynb)
      * [Ill-conditioned matrix: inverse of Hilbert matrix](Linear/Hilbert-matrix.ipynb)
      * [High-dimensional inverse problem: 1-D elliptic equation](Linear/Elliptic.ipynb)

   * Chaotic systems
       * [Chaos and butterfly effects](Chaotic/Chaos.ipynb)
       * [Lorenz63 model](Chaotic/Lorenz63.ipynb)
       * [Lorenz96 model](Chaotic/Lorenz96.ipynb)
       * [Kuramoto-Sivashinksy equation model](Chaotic/Kuramoto-Sivashinksy.ipynb)

   * Structure mechanics problems
       * Damage detection of a "bridge"
       * Consitutive modeling of a multiscale fiber-reinforced plate

   * Fluid mechanics problems
       * [2D Darcy flow](Fluid/Darcy-2D.ipynb)
       * [Navier-Stokes initial condition recovery](Fluid/Navier-Stokes.ipynb)

   * Fluid structure interaction problems
       * Piston problem
           * [Receding piston (analytical solution)](FSI-Piston/Receding-Piston-Exact.ipynb)
           * [Piston system calibration](FSI-Piston/FSI.ipynb)
       * [Airfoil damage detection during transonic buffeting](FSI-AERO/README.md)


   * Climate modeling
       * Barotropic climate model
       * Idealized general circulation model (Held-Suarez benchmark)
    
   * Other posterior distribution estimations
       * [Some nonlinear maps](Posterior/Nonlinear-Maps.ipynb)
       * [2-parameter elliptic equation](Posterior/Elliptic.ipynb)
       * [1D Darcy flow](Posterior/Darcy-1D.ipynb)

## Submit an issue
You are welcome to submit an issue for any questions related to InverseProblems. 

## Here are some research papers using InverseProblem
1. Daniel Zhengyu Huang, Tapio Schneider, and Andrew M. Stuart. "[Iterated Kalman Methodology For Inverse Problems / Unscented Kalman Inversion](https://arxiv.org/pdf/2102.01580.pdf)."

2. Daniel Zhengyu Huang, Jiaoyang Huang, Sebastian Reich, and Andrew M. Stuart. "[Efficient Derivative-free Bayesian Inference for Large-Scale Inverse Problems](https://arxiv.org/pdf/2204.04386.pdf)."

3. Shunxiang Cao, Daniel Zhengyu Huang. "[Bayesian Calibration for Large-Scale Fluid Structure Interaction Problems Under Embedded/Immersed Boundary Framework](https://arxiv.org/pdf/2105.09497.pdf)."

4. Yifan Chen, Daniel Zhengyu Huang, Jiaoyang Huang, Sebastian Reich, and Andrew M. Stuart. "[Gradient Flows for Sampling: Mean-Field Models, Gaussian Approximations
and Affine Invariance](https://arxiv.org/pdf/2302.11024.pdf)."

5. Daniel Zhengyu Huang, Jiaoyang Huang, and Zhengjiang Lin. "[Convergence Analysis of Probability Flow ODE for Score-based Generative Models](https://arxiv.org/pdf/2404.09730)."

6. Yifan Chen, Daniel Zhengyu Huang, Jiaoyang Huang, Sebastian Reich, and Andrew M. Stuart. "[Efficient, Multimodal, and Derivative-Free Bayesian Inference With Fisher-Rao Gradient Flows](https://arxiv.org/pdf/2406.17263)." 

7. Baojun Che, Yifan Chen, Zhenghao Huan, Daniel Zhengyu Huang, Weijie Wang. "[Stable Derivative Free Gaussian Mixture Variational Inference for Bayesian Inverse Problems](https://arxiv.org/pdf/2501.04259)." 
