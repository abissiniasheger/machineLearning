function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

%% Finding closest centriods
m = size(X, 1);
index = 0;
for i = 1:m
    minDistance = Inf;   
    for j = 1:K
        distance = sum((centroids(j, :) - X(i, :)).^2);
        if distance < minDistance
            minDistance = distance;
            index = j;
        end
    end
    idx(i) = index;
end
