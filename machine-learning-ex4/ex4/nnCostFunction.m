function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

y = eye(num_labels)(y,:); % y is now of size 5000 x 10
a1 = X; % a1 is of size 5000 x 400

a1 = [ones(m, 1) a1];
z2 = Theta1 * a1'; % z2 is of size 25 x 5000
a2 = sigmoid(z2); % a2 is of size 25 x 5000

a2 = [ones(m, 1) a2'];
z3 = Theta2 * a2'; % z3 is of size 10 x 5000
h = sigmoid(z3); % h (a3) is of size 10 x 5000

for i = 1:num_labels % un-regulated cost function
  J = J + 1/m * (-y(:,i)' * log(h'(:,i)) - (1 .- y(:,i))' * log(1 .- h'(:,i)));
endfor

Theta1_true = Theta1(:, 2:end); % Theta1_true is of size 25 x 400
Theta2_true = Theta2(:, 2:end); % Theta2_true is of size 10 x 25

J = J + lambda/(2*m)*(sum(sum(Theta1_true .^2)) + sum(sum(Theta2_true .^2))); % regulated cost function



D3 = h .- y'; % D3 is of size 10 x 5000
D2 = (Theta2_true' * D3) .* (sigmoidGradient(z2)); % D2 is of size 25 x 5000

Delta_1 = 0;
Delta_2 = 0;

for t = 1:m
  Delta_1 = Delta_1 .+ D2(:, t) * a1(t, :); % Delta_1 is of size 25 x 401
  Delta_2 = Delta_2 .+ D3(:, t) * a2(t, :); % Delta_2 is of size 10 x 26
endfor

Theta1_grad = 1/m * Delta_1;
Theta2_grad = 1/m * Delta_2;

Theta1_grad = [Theta1_grad(:,1) (Theta1_grad(:, 2:end) + lambda/m * Theta1(:, 2:end))];
Theta2_grad = [Theta2_grad(:,1) (Theta2_grad(:, 2:end) + lambda/m * Theta2(:, 2:end))];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
