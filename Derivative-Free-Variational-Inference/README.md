#  Derivative Free Gaussian Mixture Variational Inference (DF-GMVI)
This folder along with the Julia file [/Inversion/DF_GMVI.jl](/Inversion/DF_GMVI.jl) provides the source code for the paper "Stable Derivative Free Gaussian Mixture Variational Inference for Bayesian Inverse Problems" by Baojun Che, Yifan Chen, Zhenghao Huan, Daniel Zhengyu Huang and Weijie Wang (in alphabetical order).

The programs mentioned below, including Julia files and Jupyter notebooks, can run directly after installing the required packages and importing related programs.

## DF-GMVI algorithm
The Julia file [/Inversion/DF_GMVI.jl](/Inversion/DF_GMVI.jl) provides the source code for implementation of the proposed DF-GMVI algorithm.  

## Numerical Examples
### One-Dimensional Problems
The Jupyter notebook [Square-Map-DFGMVI.ipynb](Square-Map-DFGMVI.ipynb) provides the code for solving the one-dimensional bimodal problems with DF-GMVI algorithm introduced in subsection 5.1. These problems are realized in [Square-Map.jl](Square-Map.jl)
###  Multi-Dimensional Problems
The Jupyter notebook [MultiModal-DFGMVI.ipynb](MultiModal-DFGMVI.ipynb) provides the code for solving the multi-dimensional problems Case A ~ Case E (including both 2D sampling problems and their 100-dimensional modified versions) introduced in subsection 5.2 with DF-GMVI algorithm. These problems are realized in [MultiModal.jl](MultiModal.jl)

The Julia file [MultiModal-Comparison-2D.jl](MultiModal-Comparison-2D.jl) provides the code for solving these multi-dimensional problems Case A ~ Case E in 2D with other state-of-the-art methods, and the Julia file [MultiModal-Comparison-100D.jl](MultiModal-Comparison-100D.jl) provides the code for solving Case A in 100D with the affine-invariant MCMC method.
###  High-Dimensional Inverse Problem
The Julia file [GMGD_NS.jl](GMGD_NS.jl) provides the code for solving the high-dimensional inverse problem introduced in subsection 5.3 with DF-GMVI algorithm.
##  Quadrature Rule Comparison
The Jupyter notebook [Gaussian_Mixture_Quadrature_Rule.ipynb](Gaussian_Mixture_Quadrature_Rule.ipynb) provides the code for quadrature rule comparison mentioned in Appendix A.
