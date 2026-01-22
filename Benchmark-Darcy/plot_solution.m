%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% plots solution, u, to Poisson equation %%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% associated to theta and a 32x32 mesh %%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%input matrix theta for plotting (e.g., theta = theta_hat)
theta = input('choose theta for plotting u: theta = ');

%construct mass matrix, M_plot, for plotting
xsp = 0:0.02:1;
n = length(xsp);
Mp = zeros(n,n,33^2);
for k=1:33^2
    c = ij_to_dof_inv(k);
    for i=1:n
        for j=1:n
            Mp(i,j,k) = phi(xsp(i)-h*c(1),xsp(j)-h*c(2));
        end
    end
end
Mp = reshape(Mp,[n^2 33^2]);

%run forward solver on mean of theta
A = zeros(33^2,33^2);
for i=0:31
    for j=0:31   %build A by summing over contribution from each cell

        %find local coefficient in 8x8 grid
        grd = ij_to_grid(i,j);

        %update A by including contribution from cell (i,j)
        dof = [ij_to_dof(i,j),ij_to_dof(i,j+1), ...
               ij_to_dof(i+1,j+1),ij_to_dof(i+1,j)];
        A(dof,dof) = A(dof,dof) + theta(grd)*A_loc;
    end
end

%enforce boundary condition
A(boundaries,:) = Id(boundaries,:);
A(:,boundaries) = Id(:,boundaries);

%sparsify A
A = sparse(A);

%solve linear equation for coefficients, U
U = A\b;

%close all current plots
close all

%plot solution
figure
zs = reshape(Mp*U,n,n);
surf(xsp,xsp,zs)
xticks([0 1/8 2/8 3/8 4/8 5/8 6/8 7/8 1])
yticks([0 1/8 2/8 3/8 4/8 5/8 6/8 7/8 1])
