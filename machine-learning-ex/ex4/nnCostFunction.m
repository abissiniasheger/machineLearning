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
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
m = size(X, 1);
X = [ones(m, 1) X]; 
p = zeros(size(X, 1), 1);
eye_matrix = eye(num_labels);
y_matrix = eye_matrix(y, :);

%% Forward Propagation
a1 = X;
z2 = a1 * Theta1';
a2 = sigmoid(z2);
n = size(a2, 1);
a2 = [ones(n, 1) a2];
a3 = sigmoid(a2 * Theta2');

%% Back Propagation
d3 = a3 - y_matrix;
u = sigmoid(z2) .* (1 - sigmoid(z2));
d2 = d3 * Theta2(:, 2:end) .* u;
Delta1 = d2' * a1;
Delta2 = d3' * a2;
Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;
grad = [Theta1_grad(:) ; Theta2_grad(:)];
%%
theta1 = Theta1;
theta1(:, 1) = 0;
theta1 = (lambda / (m)) .* theta1;
theta2 = Theta2;
theta2(:, 1) = 0;
theta2 = (lambda / (m)) .* theta2;
Theta1 = Theta1(:, 2:end);
Theta2 = Theta2(:, 2:end);
theta1_reg = sum(sum(Theta1.^2));
theta2_reg = sum(sum(Theta2.^2));
%% Regularized Back Propagation
Theta1_grad = Theta1_grad + theta1;
Theta2_grad = Theta2_grad + theta2;
grad = [Theta1_grad(:); Theta2_grad(:)];

%% Unregularized cost
J = -1 * sum(diag(y_matrix * log(a3)' + (1 - y_matrix) * log((1 - a3))'))/m;

%% Regularized Cost
J = J + (lambda / (2 * m)) * (theta1_reg + theta2_reg);

end
