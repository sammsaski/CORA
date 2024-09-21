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