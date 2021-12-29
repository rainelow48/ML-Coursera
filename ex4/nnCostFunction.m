function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Recode y into newY (mxK matrix)
newY = zeros(m,num_labels);
for i=1:m,
  newY(i, y(i)) = 1;
end;

% Get h_theta(x^(i)) prediction
a1 = [ones(m,1) X]; % X: mx400, a1: mx401
a2 = [ones(m,1) sigmoid(a1*Theta1')]; % a1: mx401, Theta1: 25x401, a2: mx26
pred = sigmoid(a2*Theta2'); % a2: mx26, Theta2: 10x26, pred: mx10

J = sum(sum(-newY.*log(pred)-(1-newY).*log(1-pred)))/m;

regJ = lambda*(sum(sum(Theta1(:, 2:end).*Theta1(:, 2:end)))+sum(sum(Theta2(:, 2:end).*Theta2(:, 2:end))))/(2*m);

J = J+regJ;

% Compute NN gradient
del1 = zeros(size(Theta1));
del2 = zeros(size(Theta2));

for i = 1:m,
  % Feedforward passed
  a1 = [1 X(i, :)]; % a1: 1x401
  a2 = [1 sigmoid(a1*Theta1')]; %a2: 1x26
  pred = sigmoid(a2*Theta2'); % pred: 1x10
  
  % Find deltas
  d3 = pred - newY(i, :); % d3: 1x10
  d2 = d3*Theta2(:, 2:end).*sigmoidGradient(a1*Theta1'); % d2: 1x25
  
  % Update deltas
  del1 = del1 + d2'*a1; % del1: 25x401
  del2 = del2 + d3'*a2; % del2: 10x26
  
Theta1_grad = del1/m;
Theta2_grad = del2/m;

regTheta1 = lambda.*[zeros(size(Theta1)(1),1) Theta1(:, 2:end)]./m;
regTheta2 = lambda.*[zeros(size(Theta2)(1),1) Theta2(:, 2:end)]./m;

Theta1_grad = Theta1_grad + regTheta1;
Theta2_grad = Theta2_grad + regTheta2;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
