function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));
[noOfrows noOfcols] = size(g);

for i = 1:noOfrows
    for j = 1:noOfcols
        g(i, j) =  (1 / (1 + exp(-z(i, j)))) * (1 - (1 / (1 + exp(-z(i, j)))));
    end
end
end