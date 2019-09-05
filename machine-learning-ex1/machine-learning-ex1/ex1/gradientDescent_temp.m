m = length(y); % number of training examples
J_history = zeros(iterations, 1);

J = computeCost(X, y, theta)

  %theta = theta - sum(( (theta'*X')' - y) .* X,1)' /m *alpha
  theta = theta - (((theta'*X') - y')*X)'*alpha/m