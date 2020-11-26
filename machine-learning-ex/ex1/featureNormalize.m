function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
sizeOfX = size(X);
numberOfFeatures = sizeOfX(2);

for features = 1:numberOfFeatures
    mu(features) = mean(X(:,features));
    sigma(features) = std(X(:,features));
end

X_norm = (X - mu)/sigma;
end
