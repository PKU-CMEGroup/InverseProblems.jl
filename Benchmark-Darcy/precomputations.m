%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% do all precomputations necessary for MCMC simulations %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%define mesh width
h = 1/32;   

%define dimension of theta vector
d = 64;

%define characteristic function of unit square
S = @(x,y) heaviside(x).*heaviside(y) ...
           .*(1-heaviside(x-h)).*(1-heaviside(y-h));

%define tent function on the domain [-h,h]x[-h,h]
phi = @(x,y) ((x+h).*(y+h).*S(x+h,y+h) + (h-x).*(h-y).*S(x,y) ... 
          + (x+h).*(h-y).*S(x+h,y) + (h-x).*(y+h).*S(x,y+h))/h^2;

%define function that converts from (i,j) to 33x33 dof, and its inverse
ij_to_dof = @(i,j) 33*j+i+1;
ij_to_dof_inv = @(k) [k-1-33*floor((k-1)/33),floor((k-1)/33)];

%define function that converts from (i,j) to 8x8 grid of theta values
ij_to_grid = @(i,j) 8*floor(i/4)+floor(j/4)+1;

%construct measurement matrix, M
xs = 1/14:1/14:13/14;   %measurement points
M = zeros(13,13,33^2);
for k=1:33^2
    c = ij_to_dof_inv(k);
    for i=1:13
        for j=1:13
            M(i,j,k) = phi(xs(i)-h*c(1),xs(j)-h*c(2));
        end
    end
end
M = reshape(M,[13^2 33^2]);

%construct local overlap matrix, A_loc, and identity matrix Id
A_loc = [2/3  -1/6  -1/3  -1/6;
          -1/6  2/3  -1/6  -1/3;
          -1/3 -1/6   2/3  -1/6;
          -1/6 -1/3  -1/6   2/3];
Id = eye(33^2);

%locate boundary labels
boundaries = [ij_to_dof(0:1:32,0),ij_to_dof(0:1:32,32), ...
              ij_to_dof(0,1:1:31),ij_to_dof(32,1:1:31)];

%define RHS of FEM linear system, AU = b
b = ones(33^2,1)*10*h^2;
b(boundaries) = zeros(128,1);    %enforce boundary conditions on b

%load exact z_hat values
exact_values

%set global parameters and functions for simulation
sig = 0.05;           %likelihood standard deviation
sig_pr = 2;           %prior (log) standard deviation
sig_prop = 0.0725;    %proposal (log) standard deviation
forward_solver_ = @(theta) ... 
    forward_solver(theta,ij_to_dof,ij_to_grid,A_loc,Id,boundaries,b,M);
forward_solver_with_gradient_ = @(theta) ... 
    forward_solver_with_gradient( ... 
    theta,ij_to_dof,ij_to_grid,A_loc,Id,boundaries,b,M);
log_probability_ = @(m,z) log_probability(m,z,z_hat,sig,sig_pr);
grad_log_probability_ = @(m,z,dz) ... 
    grad_log_probability(m,z,dz,z_hat,sig,sig_pr);

save precomputations.mat
