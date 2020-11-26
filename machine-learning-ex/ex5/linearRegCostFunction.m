function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
J = 0;

grad = zeros(size(theta));

grad(1) = sum((theta' * X' - y') * X(:, 1)) / m;
for j = 2:length(grad)
    grad(j) = sum((theta' * X' - y') * X(:, j)) / m + lambda * theta(j) / m;
end
grad = grad(:);
J = sum(sum((theta' * X' - y').^2))/(2 * m) + lambda * sum(theta(2:end).^2) / (2 * m);
end
