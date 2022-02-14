clear
rng("default");

load('Data_preparation.mat')
load('RF_model.mat')
load('NB_model.mat')

% Normalising training set for NB
x_test_norm = normalize(x_test)

% Predicting with RF best model
prediction_rf = str2num(cell2mat(predict(rf_Mdl,x_test))); % Prediction of RF model for testing dataset
Conf_RFTest = confusionchart(y_test,prediction_rf,'RowSummary','row-normalized');

% Predicting with NB best model
[predicted_labels_nb_test, posterier_nb_test, cost_nb_test] = predict(nb_Mdl,x_test_norm);
Conf_NBTest = confusionchart(y_test,predicted_labels_nb_test,'RowSummary','row-normalized');

save('Testing.mat',"predicted_labels_nb_test",'prediction_rf')