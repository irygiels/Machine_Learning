function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
A=X';
% a simple implementation of feature scaling
% mi=[mean(X(:,1)); mean(X(:,2))];
% max_val=[max(X(:,1)); max(X(:,2))];
% min_val=[min(X(:,1)); min(X(:,2))];
% for i=1:m
%     for k=1:2
%         if(max_val(k)-min_val(k)~=0)
%             X(i,k)=(X(i,k)-mi(k))/std(X(:,k));
%         end
%     end
% end
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    % theta analytical approach
    %
    s1=0;
    s2=0;
    for i=1:m
        s1=s1+(alpha/m)*(theta'*A(1:2,i)-y(i))*X(i,1);
        s2=s2+(alpha/m)*(theta'*A(1:2,i)-y(i))*X(i,2);
    end
    temp1=theta(1)-s1;
    temp2=theta(2)-s2;
    theta(1)=temp1;
    theta(2)=temp2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
