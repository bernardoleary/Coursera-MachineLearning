function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % Print every 100th cost function result
    if mod(iter, 100) == 0
        computeCost(X, y, theta)
    end

    % This is the gradient descent algorithm - why didn't it work?
    %temp1 = 0;
    %temp2 = 0;
    %for q = 1:m
    %    temp1 = ( theta' * X(q,:)' - y(q) ) * X(q,1);
    %end
    %for r = 1:m
    %    temp2 = ( theta' * X(r,:)' - y(r) ) * X(r,2);
    %end
    %theta(1) = theta(1) - (alpha / m) * temp1;
    %theta(2) = theta(2) - (alpha / m) * temp2;

    % This is the vectorised version, it worked - why?
    % https://www.coursera.org/learn/machine-learning/discussions/-m2ng_KQEeSUBCIAC9QURQ
    h = X * theta;
    errors_vector = h - y;
    theta_change = ( X' * errors_vector ) * ( 1 / m ) * alpha;
    theta = theta - theta_change;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
