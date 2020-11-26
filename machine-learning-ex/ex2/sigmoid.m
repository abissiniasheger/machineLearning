function g = sigmoid(z)
%SIGMOID Compute sigmoid function
size(z)
g = zeros(size(z));
sizeofG = size(g);
iterations = sizeofG(2);
for i = 1:iterations
    g(i) = 1 / (1 + exp(-1 * z(i)));
end
end
