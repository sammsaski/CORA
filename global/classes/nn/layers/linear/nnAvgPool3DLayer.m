classdef nnAvgPool3DLayer < nnConv3DLayer
% nnAvgPool2DLayer - class for average pooling 3D layers, with
% quadratic pooling region
%
% Syntax:
%    obj = nnAvgPool3DLayer(poolSize, padding, stride, dilation, name)
%
% Inputs:
%    poolSize - dimensions of the pooling region
%    padding - padding [left top right bottom]
%    stride - step size per dimension
%    name - name of the layer, defaults to type
%
% Outputs:
%    obj - generated object
%
% References:
%    [1] T. Gehr, et al. "AI2: Safety and Robustness Certification of
%        Neural Networks with Abstract Interpretation," 2018
%    [2] Practical Course SoSe '22 - Report Martina Hinz
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: nnConv3DLayer

% Authors:       Martina Hinz, Tobias Ladner
% Written:       17-June-2022
% Last update:   01-December-2022 (combine with nnConv2DLayer)
% Last revision: 17-August-2022

% ------------------------------ BEGIN CODE -------------------------------

methods
    %constructor
    function obj = nnAvgPool3DLayer(poolSize, varargin)
        if nargin < 1
            throw(CORAerror('CORA:notEnoughInputArgs', 1))
        end
        inputArgsCheck({{poolSize, 'att', 'numeric'}})

        % construct nnConv3DLayer
        W = ones(poolSize) / prod(poolSize);
        b = 0;
        obj@nnConv3DLayer(W, b, varargin{:})

        % overwrite default values for 
        if nargin < 2
            obj.stride = poolSize;
        end
        if nargin < 4
            name = [];
        end
    end

    % compute size of ouput feature map
    function outputSize = getOutputSize(obj, imgSize)
        outputSize = getOutputSize@nnConv3DLayer(obj, imgSize);
        outputSize(end) = imgSize(end); % number of channels remain
    end
end

% internal functions ------------------------------------------------------

methods(Access=protected)
    function bias = aux_getPaddedBias(obj)
        outputSize = getOutputSize(obj, obj.inputSize);
        bias = ones(prod(outputSize), 1) * obj.b;
    end
end

end

% ------------------------------ END OF CODE ------------------------------