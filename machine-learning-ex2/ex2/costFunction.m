function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% ====================== Finds incorrect cost ====================== 
% for i = 1:rows(X)
%   J += -y(i)*log(sigmoid(theta*X(i,:)))-(1-y(i))*(1-log(sigmoid(theta*X(i,:))));
% endfor
% J = J/m;

% for i = 1:rows(X)
%   for j = 1:columns(X)
%     grad(j) += (sigmoid(theta*X(i,:))-y(i))*X(i,j);
%   endfor
% endfor
% grad = grad/m;

%  ====================== Vectorised  ====================== 
J = (sum(-y'*log(sigmoid(X*theta))-(1-y')*log(1-sigmoid(X*theta))))/m;
 
grad = ((sigmoid(X*theta)-y)'*X)/m;

% =============================================================

end
