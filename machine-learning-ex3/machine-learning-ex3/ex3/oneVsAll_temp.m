%% oneVsAll Temp


%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

%displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2a: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%

% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;


%%%%%%%%%%%%%%%%%%
% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Set Initial theta
initial_theta = zeros(n + 1, 1);

% Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 50);

% Run fmincg to obtain the optimal theta
% This function will return theta and the cost 
[theta_1] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 1), lambda)), ...
           initial_theta, options);
[theta_2] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 2), lambda)), ...
           initial_theta, options);
[theta_3] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 3), lambda)), ...
           initial_theta, options);
[theta_4] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 4), lambda)), ...
           initial_theta, options);
[theta_5] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 5), lambda)), ...
           initial_theta, options);
[theta_6] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 6), lambda)), ...
           initial_theta, options);
[theta_7] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 7), lambda)), ...
           initial_theta, options);
[theta_8] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 8), lambda)), ...
           initial_theta, options);
[theta_9] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 9), lambda)), ...
           initial_theta, options);          
[theta_0] = ...
   fmincg (@(t)(lrCostFunction(t, X, (y == 10), lambda)), ...
           initial_theta, options);           
           
all_theta = [theta_1 ...
             theta_2 ...
             theta_3 ...
             theta_4 ...
             theta_5 ...
             theta_6 ...
             theta_7 ...
             theta_8 ...
             theta_9 ...
             theta_0 ]     ;
           
           
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

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
A = sigmoid(X*all_theta);
p = max(sigmoid(X*all_theta), [], 2);
p = mod(find(A'==max(A',[],1)),size(A', 1));
pos = find(p == 0);  
p(pos,1)=size(A', 1); 
