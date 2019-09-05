clear; clc;

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

mu = mean(X,1)
sigma = std(X,1)

X = (X - mu)./sigma

X = [ones(m, 1) X];