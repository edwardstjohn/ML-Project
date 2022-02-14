clear all; clc; close all;

% Setting seed for random number generator to ensure reproducibility
rng("default");

% Importing employee data from csv
data = readtable('Employee_num.csv');

% Extracting x and y values from dataset
x = table2array(data(:,1:11));
y = table2array(data(:,12));

% Categorical version of y needed for some figures
y_cat = categorical(data.LeaveOrNot); 

% Splitting data for train and test sets (75% and 25%)
split = cvpartition(y,'Holdout',0.25,'Stratify',true);

% Setting indexes for training and test sets
train_indexes = training(split);
test_indexes = test(split);

% Creating training sets
x_train = x(train_indexes,:);
y_train = y(train_indexes,:);

% Creating test sets
x_test = x(test_indexes,:);
y_test = y(test_indexes,:);

save('Data_preparation.mat', "x_train", "y_train", "x_test", "y_test")