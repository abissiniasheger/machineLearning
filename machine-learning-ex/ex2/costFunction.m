function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.
[m, n] = size(X);
grad = zeros(size(theta));
hypothesis = sigmoid(theta' * X');
J = (-1 / m) *sum(((y' * log (hypothesis)') + (1 - y)' * log (1 - hypothesis)'));

noOfExamGrades = n;
for i = 1: noOfExamGrades
    grad(i) = (1 / m) * ((hypothesis - y') * X(:, i));
end
end