clear ; close all; clc

fprintf('Plotting Data ...\n')
data = load('ex1data1.txt');
X = data(:, 1); y = data(:, 2);
m = length(y); % number of training examples


X = [ones(m, 1), data(:,1)]; % Add a column of ones to x
theta = zeros(2, 1); % initialize fitting parameters

% Some gradient descent settings
iterations = 1500;
alpha = 0.01;

%X_1 = X(1,:)  --> 1.0000   6.1101
%theta'*X_1'  --> 0

 %((theta'*X(1,:)' - y(1))^2 ) / (2*m) for one row
 hx_i = (theta'*X')' - y;
 
 %sum(((theta'*X')' - y).^2,1)/(2*m);
 %sum((sum(theta'.*X,2) - y).^2)  / (2*m)
 %sum(((theta'*X') - y').^2,2)/(2*m)
 sum((X*theta - y).^2,1)/(2*m)
%J