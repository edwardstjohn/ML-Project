clear
rng("default");

load('Data_preparation.mat')
load('Testing.mat',"predicted_labels_nb_test",'prediction_rf')

% Storing confusion matrices from testing
ConMatrix_NBTest = confusionmat(y_test, predicted_labels_nb_test);
ConMatrix_RFTest = confusionmat(y_test, prediction_rf);

% Calculating precision for NB
% Confusion matrix needs to be transposed first before calculation
ConMatrix_nb_tp = ConMatrix_NBTest';
diagonal_nb = diag(ConMatrix_nb_tp);
row_sum_nb = sum(ConMatrix_nb_tp, 2);
precision_nb = diagonal_nb ./ row_sum_nb;
precision_nb_mean = mean(precision_nb)
% Calculating precision for RF
ConMatrix_rf_tp = ConMatrix_RFTest';
diagonal_rf = diag(ConMatrix_rf_tp);
row_sum_rf = sum(ConMatrix_rf_tp, 2);
precision_rf = diagonal_rf ./ row_sum_rf;
precision_rf_mean = mean(precision_rf)

% Calculating recall for NB
column_sum_nb = sum(ConMatrix_nb_tp, 1);
recall_nb = diagonal_nb ./ column_sum_nb';
recall_nb_mean = mean(recall_nb)
% Calculating recall for RF
column_sum_rf = sum(ConMatrix_rf_tp, 1);
recall_rf = diagonal_rf ./ column_sum_rf';
recall_rf_mean = mean(recall_rf)

% Calculating F1 Score for NB
f1_score_nb = 2*((precision_nb_mean*recall_nb_mean)/(precision_nb_mean+recall_nb_mean))
% Calculating F1 Score for RF
f1_score_rf = 2*((precision_rf_mean*recall_rf_mean)/(precision_rf_mean+recall_rf_mean))

% Calculating accuracy for NB
Accuracy_nb = [];
Calculate_accuracy_nb = 100*sum(diag(ConMatrix_NBTest))./sum(ConMatrix_NBTest(:));
Accuracy_nb = [Accuracy_nb; Calculate_accuracy_nb];
% Calculating accuracy for RF
Accuracy_rf = [];
Calculate_accuracy_rf = 100*sum(diag(ConMatrix_RFTest))./sum(ConMatrix_RFTest(:));
Accuracy_rf = [Accuracy_rf; Calculate_accuracy_rf];