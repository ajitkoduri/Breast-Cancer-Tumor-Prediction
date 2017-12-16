clear; close all; clc

%loading breast cancer data from Wisconsin
CancerData = readtable('data.csv', 'ReadVariableNames', true);
Table = CancerData;
Names = Table.Properties.VariableNames;
PredictorNames = Names;
PredictorNames(:,1:2) = [];

%creates variables for data to predict outcome
predictors = Table(:,PredictorNames);
Table(:,1) = [];
A = table2array(predictors);
%creates table of expected result
response = Table.diagnosis;
%creates a model of a quadratic Support Vector Machine
Model = fitcsvm(predictors, response, 'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', {'B'; 'M'});
%creates prediction function using Model
svmPredictFcn = @(x) predict(Model, x);
predictorExtractionFcn = @(t) t(:, predictorNames);
trainedClassifier.predictFcn = @(x) svmPredictFcn(predictorExtractionFcn(x));

%adds the model as the classification SVM
trainedClassifier.ClassificationSVM = Model;

% Perform cross-validation
partitionedModel = crossval(trainedClassifier.ClassificationSVM, 'KFold', 6);

% Compute validation accuracy
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');

% Compute validation predictions and scores
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);

%creating a vector of both true benign and malignant tumors and vector of
%predicted benign and malignant tumors

t = 1:size(validationPredictions);
y = 1:size(response);
validationPredictions = cell2mat(validationPredictions);
response = cell2mat(response);
for i = 1:size(validationPredictions)
    if validationPredictions(i) == 'B'
        t(i) = 0;
    else
        if validationPredictions(i) == 'M'
        t(i) = 1;
        end
    end
end

for i = 1:size(response)
    if response(i) == 'B'
        y(i) = 0;
    else
        if response(i) == 'M'
            y(i) = 1;
        end
    end
end

%plotting confusion matrix and seeing error rates of the SVM Classification
disp('Classification using SVM: ')
figure
plotconfusion(t,y)
title('Confusion plot of Classification using SVM')
[c, cm, ind, per] = confusion(t,y);
fprintf('%d = confusion rate \n', c * length(t))
disp('Confusion Matrix is: ')
disp(cm)
fprintf('%f = false negative rate of Benign/false positive rate of Malignant classification \n', per(1,1));
fprintf('%f = false positive rate of Benign/false negative rate of Malignant classification \n', per(1,2));
fprintf('%f = true negative rate of Benign/true positive rate of Malignant classification \n', per(1,3));
fprintf('%f = true positive rate of Benign/true negative rate of Malignant classification \n', per(1,4));
fprintf('\n');

%creating scatter plot of predicted values v. true values
%incorrectly predicted have x's in the middle of them
errors = t - y;

figure
hold on
gscatter(A(:,1), A(:,28), errors, 'kw', 'x.', 10);
gscatter(A(:,1), A(:,28), validationPredictions, 'br','o', 5);
xlabel('Radius Mean')
ylabel('Texture Mean')
title('Scatter Plot of Predictions for SVM')
hold off

%creating model for Random Forest
%partitioning data for Random Forest
disp('Classification using Random Forest: ')
validation_fraction = 0.3;
n = length(t);
a = cvpartition(n, 'HoldOut', validation_fraction);

train_ID = a.training;
test_ID = a.test;

InTrain = IsTraining(train_ID,A);
OutTrain = IsTraining(train_ID,response);
InTest = IsTraining(test_ID, A);
OutTest = IsTraining(test_ID,response);

% Train a random forest (treebagger)
ntrees = 25;
leafs = 1;
trees = TreeBagger(ntrees, InTrain, OutTrain, 'Method', 'Classification', ...
    'OOBPrediction', 'on', 'OOBPredictorImportance', 'on');

% Use trained random forest on validation dataset

VData_fit = predict(trees,InTest); 

% Use trained random forest on training dataset

TData_fit = predict(trees,InTrain);

%Creating Confusion matrixes of tree bagger
a = 1:size(TData_fit);
for i = 1:size(TData_fit)
    if TData_fit{i} == 'B'
        a(i) = 0;
    else
        if TData_fit{i} == 'M'
        a(i) = 1;
        end
    end
    if OutTrain(i) == 'B'
        b(i) = 0;
    else
        if OutTrain(i) == 'M'
            b(i) = 1;
        end
    end
end

for i = 1:size(VData_fit)
    if VData_fit{i} == 'B'
        p(i) = 0;
    else
        if VData_fit{i} == 'M'
            p(i) = 1;
        end
    end
    if OutTest(i) == 'B'
        q(i) = 0;
    else
        if OutTest(i) == 'M'
            q(i) = 1;
        end
    end
end
figure
plotconfusion(a, b)
title('Confusion Plot for Training of Random Forest')
[c, cm, ind, per] = confusion(a, b);
disp('Confusion Matrix for Training of Random Forest: ')
fprintf('%d = confusion rate \n', c * length(t))
disp('Confusion Matrix is: ')
disp(cm)
fprintf('%f = false negative rate of Benign/false positive rate of Malignant classification \n', per(1,1));
fprintf('%f = false positive rate of Benign/false negative rate of Malignant classification \n', per(1,2));
fprintf('%f = true negative rate of Benign/true positive rate of Malignant classification \n', per(1,3));
fprintf('%f = true positive rate of Benign/true negative rate of Malignant classification \n', per(1,4));
fprintf('\n');
figure
plotconfusion(p, q)
title('Confusion Plot for Validation of Random Forest')
[c, cm, ind, per] = confusion(p, q);
disp('Confusion Matrix for Validation of Random Forest: ')
fprintf('%d = confusion rate \n', c * length(t))
disp('Confusion Matrix is: ')
disp(cm)
fprintf('%f = false negative rate of Benign/false positive rate of Malignant classification \n', per(1,1));
fprintf('%f = false positive rate of Benign/false negative rate of Malignant classification \n', per(1,2));
fprintf('%f = true negative rate of Benign/true positive rate of Malignant classification \n', per(1,3));
fprintf('%f = true positive rate of Benign/true negative rate of Malignant classification \n', per(1,4));
fprintf('\n');
% Finding Out of bag error of Random Forest
err = oobError(trees);
figure
plot(err);
xlabel('Number of grown trees')
ylabel('Out of bag classification error')
title('Classification error v. Trees grown')

%Finding most important variable
Importance = trees.OOBPermutedPredictorDeltaError;
WantedNames = {...
    'radius mean', 'texture mean', 'perimeter mean', 'area mean', 'smoothness mean',...
    'compactness mean', 'concavity mean', 'concavepoints mean', 'symmetry mean',...
    'fractal dimension mean', 'radius se', 'texture se', 'perimeter se', 'area se',...
    'smoothness se', 'compactness se', 'concavity se',' concavepoints se', 'symmetry se',...
    'fractal dimension se', 'radius worst', 'texture worst', 'perimeter worst', 'area worst',...
    'smoothness worst', 'compactness worst', 'concavity worst', 'concavepoints worst', 'symmetry worst', ...
    'fractal dimension worst'};
n = length(Importance);
figure
for j = 1:n
    for i = 1:(n - j)
        if (Importance(i) > Importance(i + 1))
            d = Importance(i);
            dprime = WantedNames(i);
            Importance(i) = Importance(i + 1);
            Importance(i + 1) = d;
            WantedNames(i) = WantedNames(i + 1);
            WantedNames(i + 1) = dprime;
        end
    end
end
ylabel('Variables')
xlabel('Estimate of Importance')
title('Estimate of Importance of Variables')
yticks(1:20);
WantedNames(1:10) = [];
yticklabels(WantedNames);
Importance(1:10) = [];
hold on
for i = 1:length(Importance)
    h=barh(i,Importance(i));
    if Importance(i) < 0.45
        set(h,'FaceColor','g');
    elseif Importance(i) < 0.61
        set(h,'FaceColor','y');
    else
        set(h,'FaceColor','r');
    end
end
hold off

% Unsupervised Classification Using K means clustering


X = table2array(predictors);
%Finding how many clusters is optimal
figure
for i = 10:-1:2
    [idx,C] = kmeans(X,i,'Distance','sqEuclidean',...
    'Replicates',5);
    [silh, h] = silhouette(X, idx);
    means(i - 1) = mean(silh);
end
means

%seeing how the unsupervised learning compared to the data at K = 2
figure
plot(X(idx==1,1),X(idx==1,28),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,28),'b.','MarkerSize',12)
plot(C(:,1),C(:,28),'kx',...
     'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids For K = 2'
hold off
errors = idx - 1 - y';
figure
hold on
gscatter(X(:,1), X(:,28), errors, 'kw', 'x.', 10);
gscatter(X(:,1), X(:,28), idx, 'br','o', 5);
xlabel('Radius Mean')
ylabel('Texture Mean')
title('Scatter Plot of Predictions for K means')
hold off

figure
plotconfusion(idx' - 1, y);
title('Confusion Matrix using K-Means Clustering')

%looking at the most successful amount of clustering
[idx,C] = kmeans(X,3,'Distance','sqEuclidean',...
   'Replicates',5);
figure
[silh, h] = silhouette(X, idx);
figure;
hold on
plot(X(idx==1,1),X(idx==1,28),'r.','MarkerSize',12)
plot(X(idx==2,1),X(idx==2,28),'b.','MarkerSize',12)
plot(X(idx==3,1),X(idx==3,28),'g.','MarkerSize',12)
plot(C(:,1),C(:,28),'kx',...
    'MarkerSize',15,'LineWidth',3)
legend('Cluster 1','Cluster 2','Cluster 3', 'Centroids',...
      'Location','NW')
title 'Cluster Assignments and Centroids For K = 3'
hold off

%Unsupervised Classification using Self Organizing Map

x = X';
% Creates a Self-Organizing Map
dimension1 = 3;
dimension2 = 3;
net = selforgmap([dimension1 dimension2]);

% Train the Network
[net,tr] = train(net,x);

% Test the Network
Y = net(x);

figure, plotsomnd(net)
title('Weighted Distance Between Neurons')
figure, plotsomhits(net,x)
title('Hits for SOM')
figure, plotsompos(net,x)
title('Positions of Neurons On Data')
figure, plotsomplanes(net)
