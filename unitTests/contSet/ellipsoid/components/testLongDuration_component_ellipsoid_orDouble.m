function res = testLongDuration_component_ellipsoid_orDouble
% testLongDuration_component_ellipsoid_orDouble - unit test function of 
%    ellipsoid_orDouble
%
% Syntax:  
%    res = testLongDuration_component_ellipsoid_orDouble
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
% Written:      17-March-2021
% Last update:  ---
% Last revision:---

%------------- BEGIN CODE --------------
res = true;
nRuns = 2;
bools = [false,true];
% smaller dims since halfspaces and vertices are involved
for i=[2,5:5:10]
    for j=1:nRuns
        for k=1:2 
            try
                %%% generate all variables necessary to replicate results
                E = ellipsoid.generateRandom('Dimension',i,'IsDegenerate',bools(k));
                V = randn(dim(E),2*dim(E));
                %%%
                
                for m=1:2
                    [U,S,V] = svd(V);
                    S(1,1) = bools(m)*S(1,1);
                    V = U*S*V';
                    Eo = or(E,V);
                    
                    % check whether V are contained in Eo
                    if ~contains(Eo,V)
                        res = false;
                        break;
                    end
                end
                if ~res
                    break;
                end
            catch ME
                if strcmp(ME.identifier,'CORA:solverIssue')
                    disp('Randomly generated ellipsoids caused solver issues! Ignoring...');
                    continue;
                end
                rethrow(ME);
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

% save if failed
if ~res
    path = pathFailedTests(mfilename());
    save(path,'E','V');
end
%------------- END OF CODE --------------