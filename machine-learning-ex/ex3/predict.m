function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
X = [ones(m, 1) X]; 
p = zeros(size(X, 1), 1);

a_1 = X;
a_2 = sigmoid(Theta1 * X');
%Add the bias on a_2
n = size(a_2, 2);
a_2 = [ones(1, n); a_2];

a_3 = sigmoid(Theta2 * a_2);
for i = 1:m
   [~, loc] = max(a_3(: , i));
   p(i) = loc;
end

end
