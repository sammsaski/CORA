classdef nnConv3DLayer < nnLayer
% nnConv3DLayer - class for convolutional 3D layers

properties (Constant)
    is_refinable = false
end

properties
    W, b, stride, padding
end

methods
    % constructor
    function obj = nnConv3DLayer(varargin)
        if nargin > 5
            throw(CORAerror('CORA:tooManyInputArgs', 5))
        end

        % 1. parse input arguments: varargin -> vars
        [W, b, padding, stride, name] = aux_parseInputArgs(varargin{:});

        % 2. check correctness of input arguments
        aux_checkInputArgs(W, b, padding, stride, name)

        % 3. call super class constructor
        obj@nnLayer(name)

        % 4. assign properties
        obj.W = W;
        obj.b = b;
        obj.padding = padding;
        obj.stride = stride;
    end

    function [nin, nout] = getNumNeurons(obj)
        if isempty(obj.inputSize)
            nin = [];
            nout = [];
        else
            % Compute the number of neurons if the input size was set.
            nin = prod(obj.inputSize);
            outputSize = getOutputSize(obj, obj.inputSize);
            nout = prod(outputSize);
        end
    end

    % compute size of output feature map
    function outputSize = getOutputSize(obj, imgSize)
        in_d = imgSize(1);
        in_h = imgSize(2);
        in_w = imgSize(3);

        f_d = size(obj.W, 1);
        f_h = size(obj.W, 2);
        f_w = size(obj.W, 3);

        % padding [left, top, front, right, bottom, back]
        pad_l = obj.padding(1);
        pad_t = obj.padding(2);
        pad_f = obj.padding(3);
        pad_r = obj.padding(4);
        pad_b = obj.padding(5);
        pad_bk = obj.padding(6);

        out_d = floor((in_d - f_d + pad_f + pad_bk) / obj.stride(1)) + 1;
        out_h = floor((in_h - f_h + pad_t + pad_b) / obj.stride(2)) + 1;
        out_w = floor((in_w - f_w + pad_l + pad_r) / obj.stride(3)) + 1;
        out_c = size(obj.W, 5);
        outputSize = [out_d, out_h, out_w, out_c];
    end
end

% evaluate ------------------------------------------------------------

methods  (Access = {?nnLayer, ?neuralNetwork})
    % All dimensions (d,h,w,c) are flattened into a vector

    % numeric
    function r = evaluateNumeric(obj, input, evParams)
        % compute weight and bias
        disp("Computing weight matrix")
        Wff = aux_computeWeightMatrix(obj);
        bias = aux_getPaddedBias(obj);
        disp("Done computing weight matrix")

        % simulate using linear layer
        disp("Create linear layer")
        linl = nnLinearLayer(Wff, bias);
        disp("Done creating linear layer")
        r = linl.evaluateNumeric(input, evParams);
    end
end

% Auxiliary functions ------------------------------------------------------

methods
    function layer = convert2nnLinearLayer(obj)
        % Convert the convolutional layer to an equivalent linear layer.

        % compute weight and bias
        Wff = aux_computeWeightMatrix(obj);
        bias = aux_getPaddedBias(obj);

        layer = nnLinearLayer(Wff, bias, sprintf("%s_linear", obj.name));
    end
end

methods (Access = protected)
    % Compute weight matrix to express convolutions as matrix-vector multiplication.
    function Wff = aux_computeWeightMatrix(obj)
        checkInputSize(obj)

        % Input dimensions
        img_d_in = obj.inputSize(1);
        img_h_in = obj.inputSize(2);
        img_w_in = obj.inputSize(3);
        if length(obj.inputSize) < 4
            c_in = 1;
        else
            c_in = obj.inputSize(4);
        end

        % Output dimensions
        outputSize = getOutputSize(obj, obj.inputSize);
        img_d_out = outputSize(1);
        img_h_out = outputSize(2);
        img_w_out = outputSize(3);
        if length(outputSize) < 4
            c_out = 1;
        else
            c_out = outputSize(4);
        end

        % Initialize weight matrix
        Wff = zeros(img_d_out * img_h_out * img_w_out * c_out, img_d_in * img_h_in * img_w_in * c_in);
        disp("Done initializing Wff")
        disp(size(Wff))

        % Loop over output channels and input channels
        for k = 1:c_out % number of filters/out-channels
            for i = 1:c_in % number of in-channels
                % Get the filter for this input-output channel pair
                filter = obj.W(:, :, :, i, k);

                % Compute weight matrix for this filter
                Wf = aux_computeWeightMatrixForFilter(obj, filter, obj.inputSize(1:3));

                % Compute indices for the ith input channel and kth output channel
                % Input indices
                input_channel_indices = (i - 1) * img_d_in * img_h_in * img_w_in + (1:img_d_in * img_h_in * img_w_in);

                % Output indices
                output_channel_indices = (k - 1) * img_d_out * img_h_out * img_w_out + (1:img_d_out * img_h_out * img_w_out);

                % Add Wf to Wff
                Wff(output_channel_indices, input_channel_indices) = Wff(output_channel_indices, input_channel_indices) + Wf;
            end
        end
    end

    % Compute linear filter-matrix for a single 3D-filter to express a convolution as matrix-vector-multiplication.
    function Wf = aux_computeWeightMatrixForFilter(obj, filter, imgSize)
        % Compute output size
        [f_d, f_h, f_w] = size(filter);
        outputSize = getOutputSize(obj, obj.inputSize);
        img_d_out = outputSize(1);
        img_h_out = outputSize(2);
        img_w_out = outputSize(3);

        % Compute input size including padding
        img_d_in = imgSize(1);
        img_h_in = imgSize(2);
        img_w_in = imgSize(3);

        % padding [left, top, front, right, bottom, back]
        pad_l = obj.padding(1);
        pad_t = obj.padding(2);
        pad_f = obj.padding(3);
        pad_r = obj.padding(4);
        pad_b = obj.padding(5);
        pad_bk = obj.padding(6);

        % Padded input dimensions
        img_d_in_padded = img_d_in + pad_f + pad_bk;
        img_h_in_padded = img_h_in + pad_t + pad_b;
        img_w_in_padded = img_w_in + pad_l + pad_r;

        % Stride values
        stride_d = obj.stride(1);
        stride_h = obj.stride(2);
        stride_w = obj.stride(3);

        % Initialize weight matrix
        num_output_voxels = img_d_out * img_h_out * img_w_out;
        num_input_voxels = img_d_in * img_h_in * img_w_in;
        Wf = zeros(num_output_voxels, num_input_voxels);

        % Create padded input indices
        % padded_input = zeros(img_d_in_padded, img_h_in_padded, img_w_in_padded);
        padded_input = zeros(img_w_in_padded, img_h_in_padded, img_d_in_padded);
        % input_indices = reshape(1:num_input_voxels, [img_d_in, img_h_in, img_w_in]);
        input_indices = reshape(1:num_input_voxels, [img_w_in, img_h_in, img_d_in]);
        d_start = pad_f + 1;
        h_start = pad_t + 1;
        w_start = pad_l + 1;
        % padded_input(d_start:d_start+img_d_in-1, h_start:h_start+img_h_in-1, w_start:w_start+img_w_in-1) = input_indices;
        padded_input(w_start:w_start+img_w_in-1, h_start:h_start+img_h_in-1, d_start:d_start+img_d_in-1) = input_indices;


        % Flatten filter
        % filter_flat = filter(:);
        filter = permute(filter, [3, 1, 2]);
        filter_flat = filter(:);

        % For each output voxel, compute the corresponding indices in the input and fill Wf
        row = 1;
        for od = 1:img_d_out
            for oh = 1:img_h_out
                for ow = 1:img_w_out
                    % Compute the starting indices in the padded input
                    id_start = (od - 1) * stride_d + 1;
                    ih_start = (oh - 1) * stride_h + 1;
                    iw_start = (ow - 1) * stride_w + 1;

                    % Extract the input indices corresponding to the filter window
                    id_range = id_start:id_start+f_d-1;
                    ih_range = ih_start:ih_start+f_h-1;
                    iw_range = iw_start:iw_start+f_w-1;

                    % Get the indices in the padded input
                    % indices = padded_input(id_range, ih_range, iw_range);
                    indices = padded_input(iw_range, ih_range, id_range);

                    % Flatten indices
                    indices_flat = indices(:)';

                    % Remove zero indices (due to padding)
                    valid = indices_flat > 0;
                    indices_flat = indices_flat(valid);
                    filter_flat_valid = filter_flat(valid);

                    % Fill the row of Wf
                    Wf(row, indices_flat) = filter_flat_valid;

                    row = row + 1;
                end
            end
        end
    end

    % Pad bias so that convolution can be simulated by linear layer.
    function bias = aux_getPaddedBias(obj)
        % Expand the bias vector to output size
        outputSize = getOutputSize(obj, obj.inputSize);
        img_d_out = outputSize(1);
        img_h_out = outputSize(2);
        img_w_out = outputSize(3);
        c_out = outputSize(4);

        % Repeat the bias for each output voxel
        bias = repmat(obj.b', img_d_out * img_h_out * img_w_out, 1);
        bias = reshape(bias, [], 1);
    end
end

end

% Auxiliary functions -----------------------------------------------------

function [W, b, padding, stride, name] = aux_parseInputArgs(varargin)
    % validate input
    [W, b, padding, stride, name] = setDefaultValues( ...
        {1, 0, [0 0 0 0 0 0], [1 1 1], []}, varargin);
end

function aux_checkInputArgs(W, b, padding, stride, name)
    % check input types
    inputArgsCheck({ ...
        {W, 'att', 'numeric'}, ...
        {b, 'att', 'numeric'}, ...
        {padding, 'att', 'numeric'}, ...
        {stride, 'att', 'numeric'}, ...
    })

    % Check dimensions
    if length(size(W)) > 5
        throw(CORAerror('CORA:wrongInputInConstructor','Weight matrix has wrong dimensions.'))
    end
    if size(W, 5) ~= length(b)
        throw(CORAerror("CORA:wrongInputInConstructor",'Weight tensor and bias dimensions do not match.'))
    end
    if size(padding) ~= [2 3]
        throw(CORAerror('CORA:wrongInputInConstructor', 'Padding must be an array with 6 entries: [left, top, front, right, bottom, back]'))
    end
    if length(stride) ~= 3
        throw(CORAerror('CORA:wrongInputInConstructor', 'Stride must be an array with 3 entries: [stride_depth, stride_height, stride_width]'))
    end
end
