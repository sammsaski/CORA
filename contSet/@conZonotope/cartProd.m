function cZ = cartProd(cZ,S)
% cartProd - Returns the Cartesian product of a constrained zonotope and
%    other set representations or points
% 
% Syntax:  
%    cZ = cartProd(cZ,S)
%
% Inputs:
%    cZ - conZonotope object
%    S - contSet object
%
% Outputs:
%    cZ - conZonotope object
%
% Example: 
%    Z = [0 1 2];
%    A = [1 1]; b = 1;
%    cZ = conZonotope(Z,A,b);
%    Z = zonotope([0 1]);
%    cZcart = cartProd(cZ,Z);
%
%    figure; hold on; xlim([0.5 2.5]); ylim([-1.5 1.5]);
%    plot(cZcart,[1,2],'FaceColor','r');
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: zonotope/cartProd

% Author:       Niklas Kochdumper
% Written:      10-August-2018
% Last update:  05-May-2020 (MW, standardized error message)
% Last revision:---

%------------- BEGIN CODE --------------

% pre-processing
[res,vars] = pre_cartProd('conZonotope',cZ,S);

% check premature exit
if res
    % if result has been found, it is stored in the first entry of var
    cZ = vars{1}; return
else
    % potential re-ordering
    cZ = vars{1}; S = vars{2};
end


% first or second set is constrained zonotope
if isa(cZ,'conZonotope')

    % different cases for different set representations
    if isa(S,'conZonotope')

        % new center vector
        c = [cZ.Z(:,1); S.Z(:,1)];

        % new generator matrix
        G = blkdiag(cZ.Z(:,2:end),S.Z(:,2:end));

        % new constraint matrix
        h1 = size(cZ.A,1);
        h2 = size(S.A,1);

        m1 = size(cZ.Z,2)-1;
        m2 = size(S.Z,2)-1;

        if isempty(cZ.A)
           if isempty(S.A)
              A = []; 
           else
              A = [zeros(h2,m1),S.A];
           end
        else
           if isempty(S.A)
              A = [cZ.A,zeros(h1,m2)];
           else
              A = [[cZ.A,zeros(h1,m2)];[zeros(h2,m1),S.A]];
           end
        end

        % new constraint offset
        b = [cZ.b;S.b];

        % generate resulting constrained zonotope
        cZ = conZonotope([c,G],A,b);

    elseif isnumeric(S) || isa(S,'zonotope') || ...
           isa(S,'interval') || isa(S,'mptPolytope') || ...
           isa(S,'zonoBundle')

        cZ = cartProd(cZ,conZonotope(S));
        
    elseif isa(S,'polyZonotope') || isa(S,'conPolyZono')
        
        cZ = cartProd(polyZonotope(cZ),S);
        
    else
        % throw error for given arguments
        throw(CORAerror('CORA:noops',cZ,S));
    end

else

    % different cases for different set representations
    if isnumeric(cZ)

        cZ = cartProd(conZonotope(cZ),S);

    else
        % throw error for given arguments
        throw(CORAerror('CORA:noops',cZ,S));
    end
    
end

%------------- END OF CODE --------------