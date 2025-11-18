%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% forward solver function %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%INPUTS: 
%theta = current 8x8 parameter matrix
%lbl = cell labeling function
%A_loc = matrix of local contributions to A
%Id = Identity matrix of size 128x128
%boundaries = labels of boundary cells
%b = right hand side of linear system (AU = b)
%M = measurement matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%OUTPUTS:
%z = vector of measurements
%dz = gradient of vector of measurements with respect to m = log(theta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [z,dz] = ... 
    forward_solver_with_gradient( ... 
        theta,ij_to_dof,ij_to_grid,A_loc,Id,boundaries,b,M)

%initialize matrix A for FEM linear solve, AU = b
A = zeros(33^2,33^2);

%loop over cells to build A
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

%solve two linear equations simultaneously for z and dz
coeffs = A\[b,M'];

%get matrix U, the solution to AU = b
U = coeffs(:,1);

%get z values
z = M*U;

%now compute derivative, dz, of z with respect to theta
dz = zeros(13^2,64);
MA_inv = coeffs(:,2:end)';    %MA_inv = M*A^(-1)
for i=0:31
    for j=0:31    %build dz by summing over contribution from each cell
        %find location in 8x8 grid associated to cell (i,j)
        grd = ij_to_grid(i,j);

        %update dz by including contribution from cell (i,j)
        dof = [ij_to_dof(i,j),ij_to_dof(i,j+1), ...
               ij_to_dof(i+1,j+1),ij_to_dof(i+1,j)];
        dz(:,grd) = dz(:,grd) - MA_inv(:,dof)*(A_loc*U(dof));
    end
end

%the above gives gradient of z wrt theta; to get gradient wrt m, do:
dz = dz.*theta';
