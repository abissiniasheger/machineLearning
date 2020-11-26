function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    temp0 = theta(1) - alpha * sum(X * theta - y)/m;
    temp1 = theta(2) - alpha * sum((X * theta - y).*(X(:,2)))/m;
    
    theta(1) = temp0;
    theta(2) = temp1;

    J_history(iter) = computeCost(X, y, theta);
    J_history(iter) = computeCost(X, y, theta);
end

end
