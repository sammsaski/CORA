% Define the weights W
W = [[1 -1; 0 1]; [-1 0; 1 -1]];
W = reshape(W, [2, 2, 2]); % [depth, height, width]
W = reshape(W, [2, 2, 2, 1, 1]); % [depth, height, width, in_channels, out_channels]

% Define the bias b
b = 0;

% Define the input
in = [[1 2 3; 4 5 6; 7 8 9]; [10 11 12; 13 14 15; 16 17 18]; [19 20 21; 22 23 24; 25 26 27]];
in = reshape(in, [3, 3, 3]); % [depth, height, width]
in = permute(in, [3, 1, 2]);
n = size(in);

% Flatten the input vector (without any additional reshaping)
in_vector = in(:);

% Define stride and padding
s = [1 1 1];
p = [0 0 0; 0 0 0]; % [left top front; right bottom back]

% Create the nnConv3DLayer object
layer = nnConv3DLayer(W, b, p, s, 'newtestconv3d');

% Set the input size
layer.inputSize = [3, 3, 3, 1];

% Convert to linear layer
fc = layer.convert2nnLinearLayer();

% Compute the output
output = fc.W * in_vector + fc.b;

% Display the output
disp('Output:');
disp(output);

% Display the weight matrix
disp('Generated Weight Matrix Wff:');
disp(fc.W);


%% Example 2

% Define the weights W
W = [[1 -1; -1 1]; [0 0; 1 1]];
W = reshape(W, [2, 2, 2]); % [depth, height, width]
W = reshape(W, [2, 2, 2, 1, 1]); % [depth, height, width, in_channels, out_channels]

% Define the bias b
b = 0;

% Define the input
in = [[6 6 2; 8 4 7; 6 1 4]; [6 7 4; 0 0 1; 8 2 7]; [0 5 1; 4 1 5; 5 1 5]];
in = reshape(in, [3, 3, 3]); % [depth, height, width]
in = permute(in, [3, 1, 2]);
n = size(in);

% Flatten the input vector (without any additional reshaping)
in_vector = in(:);

% Define stride and padding
s = [1 1 1];
p = [0 0 0; 0 0 0]; % [left top front; right bottom back]

% Create the nnConv3DLayer object
layer = nnConv3DLayer(W, b, p, s, 'newtestconv3d');

% Set the input size
layer.inputSize = [3, 3, 3, 1];

% Convert to linear layer
fc = layer.convert2nnLinearLayer();

% Compute the output
output = fc.W * in_vector + fc.b;

% Display the output
disp('Output:');
disp(output);

% Display the weight matrix
disp('Generated Weight Matrix Wff:');
disp(fc.W);

%% Example 3

% Define the weights W
W = [[1 -1; -1 1]; [0 0; 1 1]];
W = reshape(W, [2, 2, 2]); % [depth, height, width]
W = reshape(W, [2, 2, 2, 1, 1]); % [depth, height, width, in_channels, out_channels]

% Define the bias b
b = 0;

% Define the input
in = zeros([3 3 3]);
in(:, :, 1) = [6 6 2; 8 4 7; 6 1 4];
in(:, :, 2) = [6 7 4; 0 0 1; 8 2 7];
in(:, :, 3) = [0 5 1; 4 1 5; 5 1 5];
% in = reshape(in, [3, 3, 3]); % [depth, height, width]
% in = permute(in, [3, 1, 2]);
% n = size(in);

% Flatten the input vector (without any additional reshaping)
% in_vector = in(:);

% Define stride and padding
s = [1 1 1];
p = [0 0 0; 0 0 0]; % [left top front; right bottom back]

% Create the nnConv3DLayer object
layer = nnConv3DLayer(W, b, p, s, 'newtestconv3d');

% Set the input size
layer.inputSize = [3, 3, 3, 1];

% Convert to linear layer
fc = layer.convert2nnLinearLayer();

output = fc.evaluateNumeric(in);

% Display the output
disp('Output:');
disp(output);

% Display the weight matrix
disp('Generated Weight Matrix Wff:');
disp(fc.W);

%% Example 4

% Define the weights W
W = zeros([2 2 2 3 1]);
W(:, :, :, 1) = reshape([[1 0; -1 1]; [0 -1; 1 0]], [2 2 2]);
W(:, :, :, 2) = reshape([[1 -1; 0 1]; [1 0; -1 1]], [2 2 2]);
W(:, :, :, 3) = reshape([[0 1; -1 0]; [1 1; 0 -1]], [2 2 2]);

% W = reshape(W, [2, 2, 2]); % [depth, height, width]
% W = reshape(W, [2, 2, 2, 1, 1]); % [depth, height, width, in_channels, out_channels]

% Define the bias b
b = 0;

% Define the input
in = zeros([3 3 3 3]);
% (h, w, d, c)
in(:, :, 1, 1) = [1 2 1; 0 1 2; 1 0 1];
in(:, :, 2, 1) = [1 1 0; 2 0 1; 1 2 0];
in(:, :, 3, 1) = [2 1 1; 1 0 2; 0 2 1];

in(:, :, 1, 2) = [2 1 0; 1 1 2; 0 2 1];
in(:, :, 2, 2) = [1 0 2; 1 1 0; 2 0 1];
in(:, :, 3, 2) = [0 1 2; 2 1 1; 1 2 0];

in(:, :, 1, 3) = [0 1 2; 1 0 1; 2 1 0];
in(:, :, 2, 3) = [1 2 1; 0 1 2; 1 0 1];
in(:, :, 3, 3) = [2 0 1; 1 2 0; 0 1 2];

% change from (h,w,d,c) to (d,h,w,c) like conv3D expects
% in = permute(in, [3 1 2 4]);

% Define stride and padding
s = [1 1 1];
p = [0 0 0; 0 0 0]; % [left top front; right bottom back]

% Create the nnConv3DLayer object
layer = nnConv3DLayer(W, b, p, s, 'newtestconv3d');

nn = neuralNetwork({layer});

% Set the input size
nn.layers{1, 1}.inputSize = [3, 3, 3, 3];

output = nn.evaluate(in);

% Convert to linear layer
% fc = layer.convert2nnLinearLayer();

% output = fc.evaluateNumeric(in);

% Display the output
disp('Output:');
disp(output);

% Display the weight matrix
% disp('Generated Weight Matrix Wff:');
% disp(fc.W);

%% Example 5

% Define the weights W
W = ones([3 3 3 3 8]);

% Define the bias b
b = zeros(8, 1);

% Define the input
in = zeros([16 32 32 3]);
% (h, w, d, c)

count = 1;
for c=1:3
    for d=1:16
        for h=1:32
            for w=1:32
                in(d,h,w,c) = count;
                count = count + 1;
            end
        end
    end
end
                
% Define stride and padding
s = [1 1 1];
p = [0 0 0; 0 0 0]; % [left top front; right bottom back]

% Create the nnConv3DLayer object
layer = nnConv3DLayer(W, b, p, s, 'newtestconv3d');

nn = neuralNetwork({layer});

% Set the input size
nn.layers{1, 1}.inputSize = [16, 32, 32, 3];

output = nn.evaluate(in);

% Convert to linear layer
% fc = layer.convert2nnLinearLayer();

% output = fc.evaluateNumeric(in);

% Display the output
disp('Output:');
disp(output);

% Display the weight matrix
% disp('Generated Weight Matrix Wff:');
% disp(fc.W);