function res = testLongDuration_halfspace_dim
% testLongDuration_halfspace_dim - unit test function of dim
%
% Syntax:  
%    res = testLongDuration_halfspace_dim
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

% Author:       Mark Wetzlinger
% Written:      27-Sep-2019
% Last update:  16-March-2021 (MW, add empty case)
% Last revision:---

%------------- BEGIN CODE --------------

nrTests = 1000;
res = true;

for i=1:nrTests
    % random dimension
    n = randi(50);
    
    % random normal vector and distance
    c = randn(n,1);
    d = randn(1);
    
    % init halfspace
    h = halfspace(c,d);
    
    % check result
    if dim(h) ~= n
        res = false; break;
    end
end

if ~res
    path = pathFailedTests(mfilename());
    save(path,'c','d','n');
end

%------------- END OF CODE --------------