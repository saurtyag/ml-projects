X = csvread('trainData.csv',0,1); 
y = csvread('trainLabels.csv',0,1);
XVal = csvread('valData.csv',0,1);
yVal = csvread('valLabels.csv',0,1);
XTest = csvread('testData_new.csv',0,1);
% XTest = csvread('testData.csv',0,1);
lambda_all = [ 0.01, 0.1, 1, 10, 100, 1000];
RMSE_Train = zeros(6,2);
RMSE_Val = zeros(6,2);
RMSE_LOOCV = zeros(6,2);

% for loop for RMSE of validation data 
for k = 1:6
    %starting the for loop for 6 lambda iterations 
    % the output will be the 6 values of RMSE, one for each lambda
    [w,b,obj,cvErrs] = ridgeReg(X,y,lambda_all(k));
    w(end)=[];
    ycap = (XVal * w) + b ;
    Error = ycap - yVal;
    RMSE_Val(k,1) = sqrt(mean(Error.^2)); 
    RMSE_Val(k,2) = lambda_all(k);
    
end

% for loop for RMSE of training and LOOCV data 
for k = 1:6
    %starting the for loop for 6 lambda iterations 
    % the output will be the 6 values of RMSE, one for each lambda
    [w,b,obj,cvErrs] = ridgeReg(X,y,lambda_all(k));
    w(end)=[];
    ycap = (X * w) + b ;
   
    
    %calculating the RMSE for training data
    Error = ycap - y;
    RMSE_Train(k,1) = sqrt(mean(Error.^2)); 
    RMSE_Train(k,2) = lambda_all(k);
    if(lambda_all(k) == 1)
        % Square error term for training data
        SquareErr = sum(Error.^2);
    end
    %calculating the RMSE for LOOCV data
    RMSE_LOOCV(k,1) = sqrt(mean(cvErrs.^2)); 
    RMSE_LOOCV(k,2) = lambda_all(k);
    
end

lambda = 1;
[w,b,obj,cvErrs] = ridgeReg(X,y,lambda);
w(end) = [];
ycap = (XTest * w) + b ;
ycapLength = length(ycap);
%changing ycap to store data according to Kaggle submission format
ycap = [zeros(ycapLength,1),ycap];
% Regularization term 
Regul_term = lambda * sum(w.^2);
for i = 1:ycapLength
    ycap(i,1) = i-1;
end

% the following line will write the predicted values in yhat to a csv file.
csvwrite('predTestLabels.csv',ycap);

% the following code extracts the best and the worst elements of the weight
% vector.
[wBest, I1] = maxk(w,10);
[wWorst, I2] = mink(w,10);

% plot code 
%taking log values 
y1 = (RMSE_Train(:,1));
plot(log10(RMSE_Val(:, 2)),y1)
title('log(λ) vs RMSE')
xlabel('log 10 of λ')
ylabel('RMSE')
hold on

y2 = (RMSE_Val(:,1));
plot(log10(RMSE_Val(:, 2)),y2)

y3 = (RMSE_LOOCV(:,1));
plot(log10(RMSE_Val(:, 2)),y3)

legend('Training','Validation','LOOCV')

hold off

% This is our main function that will calulate the ridge regression
% solution
function [w,b,obj,cvErrs] = ridgeReg(X,y,lambda)
% Xunaltered = X;    %-- this value will be used when we want to calculate
% the cvErr value.
X = X';
[numRows,numCols] = size(X);  
X = [X;ones(numCols,1)'];    
zeroMatrix =  zeros(numRows,1);
I = [eye(numRows),zeroMatrix; zeroMatrix', 0];

% weight vector
C = ((X * X') + (lambda .* I));
Cinv = pinv(C);
w = Cinv *X * y;

% bias
b = w(end);

% objective function
objTerm1 = lambda * (norm(w)^2);
summation = 0;
for k = 1:numCols
    objTerm2 = w' * X(:,k);
    objTerm3 = (b - y(k));
    summation = summation + (objTerm2 + objTerm3)^2;
end
obj = objTerm1 + summation;

% vector with validation errors
cvErrs=zeros(numCols,1);
for j = 1 : numCols
    NumPart1 = (w') * X(:,j);
    Numerator = NumPart1 - y(j);
    Denominator = 1 - ((X(:,j)'* Cinv) * X(:,j));
    cvErrs(j) = Numerator / Denominator;
end
 
end 