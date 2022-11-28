function res = test_capsule_generateRandom
% test_capsule_generateRandom - unit test function of generateRandom
%
% Syntax:  
%    res = test_capsule_generateRandom
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
% Last update:  19-May-2022 (name-value pair syntax)
% Last revision:---

%------------- BEGIN CODE --------------

% empty call
C = capsule.generateRandom();

% values for tests
n = 3;
c = [2;1;-1];
r = 3.5;

% only dimension
C = capsule.generateRandom('Dimension',n);
res = dim(C) == n;

% only center
C = capsule.generateRandom('Center',c);
res(end+1,1) = all(abs(C.c - c) < eps);

% only radius
C = capsule.generateRandom('Radius',r);
res(end+1,1) = abs(C.r - r) < eps;

% dimension and center
C = capsule.generateRandom('Dimension',n,'Center',c);
res(end+1,1) = dim(C) == n && all(abs(C.c - c) < eps);

% dimension and radius
C = capsule.generateRandom('Dimension',n,'Radius',r);
res(end+1,1) = dim(C) == n && abs(C.r - r) < eps;

% center and radius
C = capsule.generateRandom('Center',c,'Radius',r);
res(end+1,1) = all(abs(C.c - c) < eps) && abs(C.r - r) < eps;

% dimension, center, and radius
C = capsule.generateRandom('Dimension',n,'Center',c,'Radius',r);
res(end+1,1) = dim(C) == n && all(abs(C.c - c) < eps) && abs(C.r - r) < eps;


% unify results
res = all(res);

%------------- END OF CODE --------------