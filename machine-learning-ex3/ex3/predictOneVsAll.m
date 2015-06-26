function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1); % m is the number of observations on x
num_labels = size(all_theta, 1); % num_labels = 10 in this exercise

% You need to return the following variables correctly 
p_temp = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

probability_matrix = sigmoid(all_theta * X'); % probability_matrix is of size 10 x 5000

max_values_matrix = max(probability_matrix', [], 2); % max_values_matrix is of size 5000 x 1

for j = 1:m
  for i = 1:num_labels
    if probability_matrix(i,j) == max_values_matrix(j,1)
      p(j,1) = i;
    endif
  endfor
endfor

% =========================================================================


end
