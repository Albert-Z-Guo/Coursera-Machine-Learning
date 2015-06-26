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

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

x_minus_mu = zeros(size(centroids,1), size(centroids,2)); % x_minus_mu is of the same size as centroids (3 x 2)
J = zeros(size(centroids,1), 1);

for i = 1:size(X,1)
  J = zeros(K, 1); % reset J
  J_minimum = 0; % reset J_minimum
  
  for j = 1:K % j is from 1 to 3
    x_minus_mu(j, :) = X(i, :) - centroids(j, :);
  endfor
  
  for m = 1:size(centroids,2)
    J = J + x_minus_mu(:, m) .^2;
  endfor
    
  J_minimum = min(min(J)); % find J_minimum
  
  for n = 1:K % n is from 1 to 3
    if J(n, 1) == J_minimum
      idx(i) = n; % categorize the index
    endif
  endfor
endfor

% =============================================================

end

