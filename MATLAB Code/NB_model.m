clear
rng("default");

load('Data_preparation.mat')

% Normalising training set for NB as needed for optimising the 'Width'
% parameter
x_train_norm = normalize(x_train);

gsErrors_nb = [];
Accuracy_nb = [];
Time_nb = [];

time_old_nb = clock;
% Running fitcnb to create NB model
nb_Mdl = fitcnb(x_train_norm,y_train,'OptimizeHyperparameters',{'DistributionNames','Width'},'HyperparameterOptimizationOptions',struct('Optimizer','gridsearch','KFold',5));
% Calculating metrics
[predicted_labels_nb, posterier_nb, cost_nb] = predict(nb_Mdl,x_train_norm);
ConMatrix_nb = confusionmat(y_train, predicted_labels_nb);
Calculate_accuracy_nb = 100*sum(diag(ConMatrix_nb))./sum(ConMatrix_nb(:));
Accuracy_nb = [Accuracy_nb; Calculate_accuracy_nb];
gsErrors_nb = [gsErrors_nb; nb_Mdl.DistributionNames nb_Mdl.Width sum(1-(y_train == predicted_labels_nb)) / length(y_train)];
Time_nb = [Time_nb; etime(clock, time_old_nb)];
Summary_metrics_nb = [gsErrors_nb Time_nb];

% Storing best hyper-parameters
BestDN = nb_Mdl.DistributionNames;
BestWidth = nb_Mdl.Width;

save('NB_model.mat','nb_Mdl')