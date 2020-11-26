function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha
m = length(y); % number of training examples
X = [ones(m, 1), X];
J_history = zeros(num_iters, 1);
sizeOfX = size(X);
numberOfFeatures = sizeOfX(2);
temp = zeros(numberOfFeatures, 1);
for iter = 1:num_iters
    for features = 1:numberOfFeatures
            temp(features) = theta(features) - alpha * sum((X * theta - y).*(X(:,features)))/m;
    end
    theta = temp;  
    J_history(iter) = computeCostMulti(X, y, theta);
end
end
