function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 
[m, n] = size(X);
lambdaCost = zeros(1, n);
lambdaGradient =  zeros(1, n);
[J, grad] = costFunction(theta, X, y);
for i = 2:n
    lambdaCost(i) = theta(i)^2;
end
J = J + (lambda/(2 * m)) * sum(lambdaCost);
for i = 2:n
    lambdaGradient(i) = theta(i);
end
grad = grad + (lambda / m) * lambdaGradient;
end