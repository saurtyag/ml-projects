%Author Saurabh Tyagi
%Loads the mat data 
load q2_1_data.mat

c=0.1;
X = trD;
y = trLb';
Xval = valD;
yval = valLb';


%Kaggle train data
X_train = csvread('TrainFeaturesupdated.csv',1,2); 
X_train = X_train';
X_train = normalized_output(X_train);
ytrain = csvread('TrainLabelsupdated.csv',1,2);
ytrain = ytrain';

%Kaggle test data
Xtest = csvread('TestFeatures1.csv',1,1); 
Xtest = Xtest';
Xtest = normalized_output(Xtest);

%Kaggle data appended 
XVal_kaggle = csvread('ValFeaturesupdated.csv',1,2);
XVal_kaggle = transpose(XVal_kaggle);
XVal_kaggle = normalized_output(XVal_kaggle);
yVal_kaggle = csvread('ValLabelsupdated.csv',1,2);
yVal_kaggle = yVal_kaggle';
X_train_large = [X_train,XVal_kaggle];
y_train_large = [ytrain,yVal_kaggle];


% cycle - 1, when C = 0.1;
%cycle1(X,Xval,y,yval,c);

%cycle - 2, when C = 10; 
cycle2(X,Xval,y,yval,c);

%method call for one vs one classification
%one_v_one_classifier(X_train_large,Xtest,y_train_large,0.1);



function [alpha,f_new,H] = return_alpha(X,y,c)
[~,k] = size(X);																	
    %k = num_samples
    A = [];
    b = [];
    Aeq = y;
    beq = 0;
    lb = zeros(k,1);
    ub = c*ones(k,1);
    f = -ones(k,1);
    
    H = (y' * y) .* (X'*X);
[alpha,f_new]=quadprog(H,f,A,b,Aeq,beq,lb,ub);
end

function [w,bias,num_sp] = return_w_b(alpha,X,y)
% code for parameter vector
w = X * (alpha .* y');

% code for finding the bias
[Xsp, Ysp, num_sp] = return_SV_values(alpha,X,y);
bias = mean((Ysp - (w' * Xsp)));
end

function [Xsp, Ysp, num_sp] = return_SV_values(alpha,X,y)
indices = find(alpha > 1e-3);
[num_sp,~] = size(indices);  
Xsp = X(:,indices);
Ysp = y(:,indices);
end

% we will pass val or test data to this function 
% and recieve the classification predictions.
function [preds] = return_preds(X,w,b)
    svm = (w' * X) + b;    
    [~,numCols] = size(svm);
    preds = zeros(1,numCols);
    for i = 1:numCols
        if svm(1,i) > 0
            preds(1,i) = 1;
        else
            preds(1,i) = -1;
        end
    end
end

% This function will return the accuracy percentage for that binary SVM <w,b>.
function [percentage_accuracy] = return_accuracy(y,preds)
    [~,numCols] = size(y);
    counter = 0;
    for index = 1:numCols
        if preds(1,index) == y(1,index)
            counter = counter + 1;
        end
    end
    
    percentage_accuracy = (counter/numCols) * 100;
end

% This function will return the objective function
function [obj_func_val] = return_objective(H,alpha ,f)
    term1 = 0.5 * alpha' * H * alpha;
    term2 = f * alpha;
    obj_func_val =  term1 + term2;
end

% This method will calculate the confusion matrix and print the confusion
% chart.
function [] = print_confusion_chart(y,preds)
    C = confusionmat(y,preds);
    confusionchart(C);
end

function [] = one_v_one_classifier(X, Xtest,y,c)
    [w12, bias12] = pairwise(X,y,c,1, 2);
    preds12 = poller_data(Xtest, w12, bias12, 1, 2);
    [w13, bias13] = pairwise(X,y,c,1, 3);
    preds13 = poller_data(Xtest, w13, bias13, 1, 3);
    [w14, bias14] = pairwise(X,y,c,1, 4);
    preds14 = poller_data(Xtest, w14, bias14, 1, 4);
    [w23, bias23] = pairwise(X,y,c,2, 3);
    preds23 = poller_data(Xtest, w23, bias23, 2, 3);
    [w24, bias24] = pairwise(X,y,c,2, 4);
    preds24 = poller_data(Xtest, w24, bias24, 2, 4);
    [w34, bias34] = pairwise(X,y,c,3, 4);
    preds34 = poller_data(Xtest, w34, bias34, 3, 4);
    [~,numColsPred] = size(preds12);
    prediction_arr = zeros(1,numColsPred);
    % polling and classification.
    predictions_matrix = [preds12;preds13;preds14;preds23;preds24;preds34];
    for pred_index = 1:numColsPred
        tempArr = [preds12(1,pred_index),preds13(1,pred_index),preds14(1,pred_index),preds23(1,pred_index),preds24(1,pred_index),preds34(1,pred_index)];
        prediction_arr(1,pred_index) = mode(tempArr);
    end
    prediction_arr = prediction_arr';
    csvwrite('predictions.csv',prediction_arr);
end

% In this function we will implement the pairwise classification based on input class.
function [w, bias] = pairwise(X,y,c,class1, class2)
     class1_elements = find(y == class1);
     class2_elements = find(y == class2);
     X1 = X(:,class1_elements);
     y1 = y(1,class1_elements);
     y1(1,:) = 1;
     X2 = X(:,class2_elements);
     y2 = y(1,class2_elements);
     y2(1,:) = -1;
     X = [X1 X2];
     y = [y1 y2];
    [alpha, ~, ~] = return_alpha(X,y,c);
    [w, bias, ~] = return_w_b(alpha,X,y);
end

% This function performs the polling data collection.
function [preds] = poller_data(X, w, b, class1, class2)
    preds = return_preds(X,w,b);
    [~,numCols] = size(preds);
    for i=1:numCols
        if eq(preds(1,i),1)
            preds(1,i) = class1;
        else 
            preds(1,i) = class2;
        end
    end
end

function [X] = normalized_output(X)
    X = normalize(X,2);
end

function [] = cycle1(X,Xval,y,yval,c)
    [alpha, f_new, H] = return_alpha(X,y,c);
    [w, bias, num_sp] = return_w_b(alpha,X,y);
    preds = return_preds(Xval,w,bias);
    percentage_accuracy = return_accuracy(yval,preds);
    obj_func_val = return_objective(H,alpha ,f_new);
    print_confusion_chart(yval,preds);
end

function [] = cycle2(X,Xval,y,yval,c)
    c=10;
    [alpha, f_new, H] = return_alpha(X,y,c);
    [w, bias, num_sp] = return_w_b(alpha,X,y);
    preds = return_preds(Xval,w,bias);
    percentage_accuracy = return_accuracy(yval,preds);
    obj_func_val = return_objective(H,alpha ,f_new);
    print_confusion_chart(yval,preds);
end