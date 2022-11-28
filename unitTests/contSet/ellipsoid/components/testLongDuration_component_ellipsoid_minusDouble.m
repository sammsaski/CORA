function res = testLongDuration_component_ellipsoid_minusDouble
% testLongDuration_component_ellipsoid_minusDouble - unit test function of
%    testLongDuration_ellipsoid_minusDouble
%
% Syntax:  
%    res = testLongDuration_component_ellipsoid_minusDouble
%
% Inputs:
%    -
%
% Outputs:
%    res - boolean 
%
% Example: 
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% Author:       Victor Gassmann
% Written:      18-March-2021
% Last update:  ---
% Last revision:---

%------------- BEGIN CODE --------------
res = true;
nRuns = 5;
bools = [false,true];
% smaller dims since halfspaces and vertices are involved
for i=10:5:15
    for j=1:nRuns
        for k=1:2 
            %%% generate all variables necessary to replicate results
            E = ellipsoid.generateRandom('Dimension',i,'IsDegenerate',bools(k));
            % generate points randomly
            V = randn(dim(E),2*dim(E));
            %%%
            for m=1:2
                [U,S,V] = svd(V);
                S(1,1) = bools(m)*S(1,1);
                V = U*S*V';
                Eo = minus(E,V);
                Ei = minus(E,V,'inner');
                Eres = ellipsoid(E.Q,E.q-sum(V,2));
                % check if the same
                if ~(Eres==Eo) || ~(Eres==Ei)
                    res = false;
                    break;
                end
            end
            if ~res
                break;
            end
        end
        if ~res
            break;
        end
    end
    if ~res
        path = pathFailedTests(mfilename());
        save(path,'E','V');
        break;
    end
end
%------------- END OF CODE --------------