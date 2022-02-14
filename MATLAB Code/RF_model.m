clear
rng("default");

load('Data_preparation.mat')

% Defining parameters
numTree = [50 75 100 150];
numPredictorsToSample = [1 3 5 7 9];

gsErrors_rf = [];
Accuracy_rf = [];
Time_rf = [];
for i=1:length(numTree)
    for o=1:length(numPredictorsToSample)
        time_old_rf = clock;
%         Running TreeBagger to create RF model
        rf_Mdl = TreeBagger(numTree(i),x_train,y_train,'SampleWithReplacement','on','Method','Classification','OOBPredictorImportance','on','MinLeafSize',10,...
            'MaxNumSplits',10,'NumPredictorsToSample',numPredictorsToSample(o));
%         Calculating metrics
        [predicted_labels_rf, scores_rf] = str2num(cell2mat(oobPredict(rf_Mdl)));
        ConMatrix_rf = confusionmat(y_train, predicted_labels_rf);
        Calculate_accuracy_rf = 100*sum(diag(ConMatrix_rf))./sum(ConMatrix_rf(:));
        Accuracy_rf = [Accuracy_rf; Calculate_accuracy_rf];
        gsErrors_rf = [gsErrors_rf; numTree(i) numPredictorsToSample(o) sum(1-(y_train == predicted_labels_rf)) / length(y_train)];
        Time_rf = [Time_rf; etime(clock, time_old_rf)];
        Summary_metrics_rf = [gsErrors_rf Time_rf];
    end
end

% Storing best hyper-parameters
BestNPS = rf_Mdl.NumPredictorsToSample;
BestTree = rf_Mdl.NumTrees;

save('RF_model.mat','rf_Mdl')