function p = randPoint(E,varargin)
% randPoint - generates a random point within an ellipsoid
%
% Syntax:  
%    p = randPoint(E)
%    p = randPoint(E,N)
%    p = randPoint(E,N,type)
%    p = randPoint(E,N,'gaussian',pr)
%
% Inputs:
%    E - ellipsoid object
%    N - number of random points
%    type - type of the random point ('standard', 'gaussian', or 'extreme')
%    pr - probability that a value is within the set (only type = 'gaussian')
%
% Outputs:
%    p - random point in R^n
%
% Example: 
%    E = ellipsoid([9.3 -0.6 1.9;-0.6 4.7 2.5; 1.9 2.5 4.2]);
%    p = randPoint(E);
% 
%    figure; hold on;
%    plot(E);
%    scatter(p(1,:),p(2,:),16,'r','filled');
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: interval/randPoint

% Author:        Victor Gassmann
% Written:       18-March-2021
% Last update:   25-June-2021 (MP, add type gaussian)
%                19-August-2022 (MW, integrate standardized pre-processing)
% Last revision: ---

%------------- BEGIN CODE --------------

% pre-processing
[res,vars] = pre_randPoint('ellipsoid',E,varargin{:});

% check premature exit
if res
    % if result has been found, it is stored in the first entry of vars
    p = vars{1}; return
else
    % assign variables
    E = vars{1}; N = vars{2}; type = vars{3}; pr = vars{4};
    % sampling with 'gaussian' is done in contSet method: see below
end

% 'all' vertices not supported
if ischar(N) && strcmp(N,'all')
    throw(CORAerror('CORA:notSupported',...
        "Number of vertices 'all' is not supported for class ellipsoid."));
end

% ellipsoid is just a point
if rank(E)==0
    % replicate point N times
    p = repmat(E.q,1,N);
    return;
end

% read center
c = E.q;
% shift center
E = E + (-c);

% compute rank and dimension
r = rank(E);
n = dim(E);

% determine degeneracy: if so, project on proper subspace (via SVD)
n_rem = n-r;
[T,~,~] = svd(E.Q);
E = T'*E;
E = project(E,1:r);
G = inv(sqrtm(E.Q));
E = G*E;


% generate different types of extreme points
if strcmp(type,'gaussian')
    pt = randPoint@contSet(E,N,type,pr);
    
    % stack again, backtransform and shift
    p = T*[inv(G)*pt;zeros(n_rem,N)] + c;
    
elseif strcmp(type,'standard')
    % generate points uniformely distributed (with N -> infinity)
    % on the unit hypersphere
    X = randn(dim(E),N);
    pt = X./sqrt(sum(X.^2,1));
    
    S = 2*rand(1,N)-1;
    pt = S.*pt;
    
    % stack again, backtransform and shift
    p = T*[inv(G)*pt;zeros(n_rem,N)] + c;

elseif strcmp(type,'extreme')
    pt = boundary(E,N);
    
    % stack again, backtransform and shift
    p = T*[inv(G)*pt;zeros(n_rem,N)] + c;

end

%------------- END OF CODE --------------