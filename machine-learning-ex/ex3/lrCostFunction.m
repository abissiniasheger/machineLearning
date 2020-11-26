function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
m = length(y); % number of training examples
hypothesis = sigmoid(theta' * X');

J = (-1 / m) *sum(((y' * log (hypothesis)') + (1 - y)' * log (1 - hypothesis)'));

temp = theta;
temp(1) = 0;

J = J + (lambda/(2 * m)) * sum(temp.^2);

grad = (1 / m) * (hypothesis - y') * X + (lambda / m) * (temp)';
grad = grad';
end
