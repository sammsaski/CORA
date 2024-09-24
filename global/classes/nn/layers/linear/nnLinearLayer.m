classdef nnLinearLayer < nnLayer
% nnLinearLayer - class for linear layers
%
% Syntax:
%    obj = nnLinearLayer(W, b)
%    obj = nnLinearLayer(W, b, name)
%
% Inputs:
%    W - weight matrix
%    b - bias column vector
%    name - name of the layer, defaults to type
%
% Outputs:
%    obj - generated object
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork

% Authors:       Tobias Ladner
% Written:       28-March-2022
% Last update:   23-November-2022 (polish)
%                14-December-2022 (variable input tests, inputArgsCheck)
% Last revision: 10-August-2022 (renamed)

% ------------------------------ BEGIN CODE -------------------------------

properties
    W                       % weight matrix
    b                       % bias
end

properties (Constant)
    is_refinable = false    % whether the layer is refineable
end

methods
    % constructor
    function obj = nnLinearLayer(W, varargin)
        % parse input
        [b, name] = setDefaultValues({0, []}, varargin);
        inputArgsCheck({ ...
            {W, 'att', 'numeric'}; ...
            {b, 'att', 'numeric'}; ...
        })

        % check dimensions
        if length(b) == 1
            b = b * ones(size(W, 1), 1);
        end
        if ~all(size(b, 1) == size(W, 1))
           throw(CORAerror('CORA:wrongInputInConstructor', ...
               'The dimensions of W and b should match.'));
        end
        if size(b, 2) ~= 1
           throw(CORAerror('CORA:wrongInputInConstructor', ...
               "Second input 'b' should be a column vector."));
        end

        % call super class constructor
        obj@nnLayer(name)

        obj.W = double(W);
        obj.b = double(b);
    end

    function outputSize = getOutputSize(obj, inputSize)
        outputSize = [size(obj.W, 1), 1];
    end

    function [nin, nout] = getNumNeurons(obj)
        nin = size(obj.W, 2);
        nout = size(obj.W, 1);
    end
end

% evaluate ----------------------------------------------------------------

methods (Access = {?nnLayer, ?neuralNetwork})
    
    % numeric
    function r = evaluateNumeric(obj, input, evParams)
        % when using conv3d layer with c_in==1 || conv2d layer with c_in==3
        if length(size(input)) == 3
            % we need to flatten the input because * doesn't support
            % n-d matrix mult when n > 2
            input = permute(input, [2, 1, 3]); % to preserve order
            input = input(:); % flatten
        elseif length(size(input)) == 4
            % input = permute(input, [2, 1, 3, 4]); % this is wrong for some reason
            
            % This is what was running when I found [1, 3, 2, 4]
            % input = permute(input, [3, 1, 2, 4]); % this works for example 5
            
            % input = permute(input, [3, 2, 1, 4]);

            % THIS WAS SO CLOSE
            % input = permute(input, [3, 2, 4, 1]);
            input = permute(input, [3, 2, 1, 4]);

            input = input(:);
        end
        r = obj.W * input + obj.b;
    end

    % sensitivity
    function S = evaluateSensitivity(obj, S, x, evParams)
        S = S * obj.W;
    end

    % zonotope/polyZonotope
    function [c, G, GI, E, id, id_, ind, ind_] = evaluatePolyZonotope(obj, c, G, GI, E, id, id_, ind, ind_, evParams)
        c = obj.W * c + obj.b;
        G = obj.W * G;
        GI = obj.W * GI;
    end

    % taylm
    function r = evaluateTaylm(obj, input, evParams)
        r = obj.W * input + obj.b;
    end

    % conZonotope
    function [c, G, C, d, l, u] = evaluateConZonotope(obj, c, G, C, d, l, u, options, evParams)
        c = obj.W * c + obj.b;
        G = obj.W * G;
    end
end

end

% ------------------------------ END OF CODE ------------------------------
