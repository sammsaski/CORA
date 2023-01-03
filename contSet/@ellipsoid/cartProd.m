function E = cartProd(E,S,varargin)
% cartProd - returns an over-approximation for the Cartesian product 
%    between two ellipsoids
%
% Syntax:  
%    E = cartProd(E,S)
%    E = cartProd(E,S,type)
%
% Inputs:
%    E - ellipsoid object
%    S - contSet object
%    type - outer approximation ('outer') or inner approximation ('inner')
%
% Outputs:
%    E - ellipsoid object
%
% Example: 
%    E1 = ellipsoid.generateRandom('Dimension',2);
%    E2 = ellipsoid.generateRandom('Dimension',2);
%    E = cartProd(E1,E2); 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: none

% Author:       Victor Gassmann
% Written:      19-March-2021
% Last update:  02-June-2022 (VG: handle empty case)
% Last revision:---

%------------- BEGIN CODE --------------

% pre-processing
[res,vars] = pre_cartProd('ellipsoid',E,S,varargin{:});

% check premature exit
if res
    % if result has been found, it is stored in the first entry of var
    E = vars{1}; return
else
    % assign variables
    E = vars{1}; S = vars{2}; type = vars{3};
end

% currently only 'outer' supported
if any(strcmp(type,{'inner','exact'}))
    throw(CORAerror('CORA:notSupported',...
        'The function ''cartProd'' supports only type = ''outer'' for ellipsoid objects.'));
end


if strcmp(type,'outer')
    
    if isa(E,'ellipsoid')
        
        if isa(S,'ellipsoid')
            % ellipsoid-ellipsoid case

            % Cartesian product of interval over-approximations
            IntE = cartProd(interval(E),interval(S));
            r = rad(IntE);
            q = center(IntE);
            TOL = min(E.TOL,S.TOL);
            
            % extract degenerate dimensions
            ind_d = withinTol(r,zeros(size(r)),TOL);
            n_nd = sum(~ind_d);
            
            % construct final ellipsoid
            E = ellipsoid(n_nd*diag(r.^2),q);
            return

        elseif isnumeric(S)
            % ellipsoid-numeric case
    
            E = ellipsoid(blkdiag(E.Q,zeros(length(S))), [E.q;S]);
            return
    
        end

    elseif isnumeric(E) && isa(S,'ellipsoid')
        % numeric-ellipsoid case

        E = ellipsoid(blkdiag(zeros(length(E)),S.Q), [E;S.q]);
        return

    end
end

% all other cases: throw error
throw(CORAerror('CORA:noops',E,S,type));

%------------- END OF CODE --------------