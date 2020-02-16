%Author Saurabh Tyagi
% The focus will be on making methods, based on the previous file 
X = csvread('TrainFeaturesupdated.csv',1,2); 
XVal = csvread('ValFeaturesupdated.csv',1,2);
y = csvread('TrainLabelsupdated.csv',1,2);
yVal = csvread('ValLabelsupdated.csv',1,2);
% Convert X into k x n format right from the start
X = transpose(X);
XVal = transpose(XVal);
XVal = normalized_output(XVal);
X_large = [X XVal];
y_large = [y; yVal];
% theta has been defined to represent the coefficients of 3 classes
% We will need 3 thetas and 4 probabilities to complete our processing
theta = zeros(513,3);  
eta0 = 0.1;
eta1 = 1;
max_epoch = 1000;
delta = 0.00001;
m = 16;
k=4;
%[theta_modified] = sgdLogisticRegression(X_large,y_large,eta0,eta1,max_epoch,delta,m,k,theta);
[theta_train] = sgdLogisticRegression_train(X,y,eta0,eta1,max_epoch,delta,m,k,theta);
Validation_preds = predictor_val(theta_train,k);
theta = zeros(513,3);  
[theta_val] = sgdLogisticRegression_val(XVal,yVal,eta0,eta1,max_epoch,delta,m,k,theta);
Train_preds = predictor_train(theta_val,k);
plot_confusion_matrix_val(Validation_preds,yVal,k);
plot_confusion_matrix_train(Train_preds,y,k);

function [theta] = sgdLogisticRegression_train(X,y,eta0,eta1,max_epoch,delta,m,k,theta)
    %Implementing the algorithm exactly as per specification
    %theta_new = ones(513,3);
    [~,numCols] = size(X);
    %added an extra rows of ones to the input features to create yield Xbar
	X_n = 	normalized_output(X);
    Xbar = [X_n;ones(numCols,1)'];
    %creating copies so that they can be manipulated to yield the elements 
    XCopy = Xbar;
    ycopy = y;
    L_thetaOld = 100;
    %beginning the loop for max epoch 
    for epoch = 1:max_epoch
        eta = eta0/(eta1+epoch);
        permutation = randperm(numCols);
        num_batches = floor(numCols/m);
        for batch = 1:num_batches
            [Xbatch,ybatch,permutation_new] = extractFromPermutation(permutation,XCopy,ycopy,m);
            permutation = permutation_new;
            [differential] = delLbyDelTheta(m,Xbatch,ybatch,theta,k);
            theta = theta - (eta * differential); 
        end
        [L_thetaNew] = loss(XCopy,ycopy,theta,numCols);
        fprintf('At epoch %d = %d \n',epoch,L_thetaNew);
        if L_thetaNew > ((1-delta)*L_thetaOld) && L_thetaNew < L_thetaOld
            break
        else
            L_thetaOld = L_thetaNew;
        end   
    end
    
end
function [theta] = sgdLogisticRegression_val(X,y,eta0,eta1,max_epoch,delta,m,k,theta)
    %Implementing the algorithm exactly as per specification
    %theta_new = ones(513,3);
    [~,numCols] = size(X);
    %added an extra rows of ones to the input features to create yield Xbar
	X_n = 	normalized_output(X);
    Xbar = [X_n;ones(numCols,1)'];
    %creating copies so that they can be manipulated to yield the elements 
    XCopy = Xbar;
    ycopy = y;
    L_thetaOld = 100;
    %beginning the loop for max epoch 
    for epoch = 1:max_epoch
        eta = eta0/(eta1+epoch);
        permutation = randperm(numCols);
        num_batches = floor(numCols/m);
        for batch = 1:num_batches
            [Xbatch,ybatch,permutation_new] = extractFromPermutation(permutation,XCopy,ycopy,m);
            permutation = permutation_new;
            [differential] = delLbyDelTheta(m,Xbatch,ybatch,theta,k);
            theta = theta - (eta * differential); 
        end
        [L_thetaNew] = loss(XCopy,ycopy,theta,numCols);
        fprintf('At epoch %d = %d \n',epoch,L_thetaNew);
        if L_thetaNew > ((1-delta)*L_thetaOld) && L_thetaNew < L_thetaOld
            break
        else
            L_thetaOld = L_thetaNew;
        end   
    end
    
end
function [Xbatch,ybatch,permutation_new] = extractFromPermutation(permutation,Xcopy,ycopy,m)
		[numRows,~] = size(Xcopy);
		Xbatch = zeros(numRows,m);
		ybatch = zeros(m,1);
		for perm = 1:m
		% Put corresponding x, in Xbatch
		% Put corresponding y, in ybatch
		% Remove corresponding element from permutation
			permutation_cycle = permutation(perm);
			Xbatch(:,perm) = Xcopy(:,permutation_cycle);
			ybatch(perm) = ycopy(permutation_cycle);
			
		end 
		
		for perm = 1:m
            if length(permutation) > perm
                permutation(perm) = [];
            end
		end
		
		permutation_new = permutation;
end

% For each image calculate 3 probabilities. 
% For first 3 probabilities, calculate the equation 8
% Store the three vectors into the a new matrix. 
% For each iteration, add these values up.
% Multiply this sum with (-1/m).
function [differential] = delLbyDelTheta(m,Xbatch,ybatch,theta,k)
		[numRows, ~] = size(ybatch);
		[numXbatchRows,~] = size(Xbatch);
		ans_storage_parent = zeros(numXbatchRows,k-1);
		for row = 1:numRows
			[prob] = probability(Xbatch(:,row),theta,k);
			ans_storage_temp = zeros(numXbatchRows,k-1);
			for c = 1:k-1
				if  eq(ybatch(row),c)
					term1 = 1;
				else 
					term1 = 0;
				end
				term2 = prob(c);
				temp_val = (term1 - term2) * Xbatch(:,k);
				ans_storage_temp(:,c) = temp_val;
			end
			ans_storage_parent = ans_storage_parent + ans_storage_temp;
		end
		differential = (-1/m)*ans_storage_parent;
end

function [lossValue] = loss(X,y,theta,n)
		[~,numThetaCols] = size(theta);
		k = numThetaCols+1;
        logsum = 0;
        for i=1:n
                [prob] = probability(X(:,i),theta,k);
                 logsum = logsum + log(prob(y(i)));
        end
		lossValue = (-1/n)*logsum;		
end

function [p] = probability(X,theta,k)
    p = zeros(k,1);
    denominator = 1;
    denominator = denominator + sum(exp(transpose(theta)*X));
	for i = 1:k-1
		numerator = exp(transpose(theta(:,i))*X);
		p(i) = numerator/denominator;
	end
	p(k) = 1/denominator;
end 

function [prediction] =  predictor_val(theta,k)
	XTest = csvread('ValFeaturesupdated.csv',1,2);
    XTest = transpose(XTest);
    [~,numCols] = size(XTest);
    
    XTest = normalized_output(XTest);
    XTest = [XTest;ones(numCols,1)'];

	prediction = zeros(numCols,1);
	for j = 1:numCols
		[p] = probability(XTest(:,j),theta,k);
		[~,p_max_index]=max(p);
		prediction(j) = p_max_index;
    end
end

function [prediction] =  predictor_train(theta,k)
	XTest = csvread('TrainFeaturesupdated.csv',1,2); 
    XTest = transpose(XTest);
    [~,numCols] = size(XTest);
    
    XTest = normalized_output(XTest);
    XTest = [XTest;ones(numCols,1)'];

	prediction = zeros(numCols,1);
	for j = 1:numCols
		[p] = probability(XTest(:,j),theta,k);
		[~,p_max_index]=max(p);
		prediction(j) = p_max_index;
    end
end

function [percentage] =  validator_train(theta,k)
	XVal = csvread('TrainFeaturesupdated.csv',1,2); 
    yVal = csvread('TrainLabelsupdated.csv',1,2);
    XVal = transpose(XVal);
    [~,numCols] = size(XVal);
    XVal = normalized_output(XVal);
    XVal = [XVal;ones(numCols,1)'];
	prediction = zeros(numCols,1);
	for j = 1:numCols
		[p] = probability(XVal(:,j),theta,k);
		[~,p_max_index]=max(p);
		prediction(j) = p_max_index;
    end
    counter = 0;
    for j = 1:numCols
        if eq(prediction(j),yVal(j))
            counter = counter + 1;
        end
    end    
    percentage = (counter/numCols);
    percentage = round(percentage,3);
    fprintf('Accuracy on data is : %d', percentage);
end

function [percentage] =  validator_val(theta,k)
	XVal = csvread('ValFeaturesupdated.csv',1,2); 
    yVal = csvread('ValLabelsupdated.csv',1,2);
    XVal = transpose(XVal);
    [~,numCols] = size(XVal);
    XVal = normalized_output(XVal);
    XVal = [XVal;ones(numCols,1)'];
	prediction = zeros(numCols,1);
	for j = 1:numCols
		[p] = probability(XVal(:,j),theta,k);
		[~,p_max_index]=max(p);
		prediction(j) = p_max_index;
    end
    counter = 0;
    for j = 1:numCols
        if eq(prediction(j),yVal(j))
            counter = counter + 1;
        end
    end    
    percentage = (counter/numCols);
    percentage = round(percentage,3);
    fprintf('Accuracy on data is : %d', percentage);
end

function [X] = normalized_output(X)
    X = normalize(X,2);
end

function [] = plotting_method_accuracy(Accuracy_store_train,Accuracy_store_val)
    figure(2)
    trailing_zeros_1 = find(Accuracy_store_train, 1, 'last');
    trailing_zeros_2 = find(Accuracy_store_val, 1, 'last');
    Accuracy_store_train = Accuracy_store_train(1:trailing_zeros_1);
    Accuracy_store_val = Accuracy_store_val(1:trailing_zeros_2);
    [numRows_train,~] = size(Accuracy_store_train); 
    [numRows_val,~] = size(Accuracy_store_val); 
    x1 = 1:numRows_train;
    x2 = 1:numRows_val;
    y1 = (Accuracy_store_train(:,1));
    y2 = (Accuracy_store_val(:,1));
    plot(x1,y1)
    
    hold on
    
    plot(x2,y2)
    title('Values of accuracy vs epoch')
    xlabel('epoch')
    ylabel('Accuracy')
    legend('Training','Validation')
    
    hold off
end

function [] =  plotting_method_Ltheta(L_theta_store_train,L_theta_store_val)
    figure(1)
    trailing_zeros_1 = find(L_theta_store_train, 1, 'last');
    trailing_zeros_2 = find(L_theta_store_val, 1, 'last');
    L_theta_store_train = L_theta_store_train(1:trailing_zeros_1);
    L_theta_store_val = L_theta_store_val(1:trailing_zeros_2);
    [numRows_train,~] = size(L_theta_store_train); 
    [numRows_val,~] = size(L_theta_store_val); 
    x1 = 1:numRows_train;
    x2 = 1:numRows_val;
    y1 = (L_theta_store_train(:,1));
    y2 = (L_theta_store_val(:,1));
    plot(x1,y1)
    
    hold on
    
    plot(x2,y2)
    title('Values of L(theta) vs epoch')
    xlabel('epoch')
    ylabel('L(\theta)')
    legend('Training','Validation')
    
    hold off

end

function [] = plot_confusion_matrix_val(Validation_preds,yVal,k)
    confusion = zeros(k,k);
    [numRows,~] =size(yVal);
    for i = 1:k
       for j = 1:k
           count = 0;
           for m = 1:numRows
               if eq(yVal(m),i) && eq(Validation_preds(m),j)
                  count = count+1; 
               end
           end
           confusion(i,j) =count;
       end
    end
    confusion 
end

function [] = plot_confusion_matrix_train(Train_preds,y,k)
    confusion = zeros(k,k);
    [numRows,~] =size(y);
    for i = 1:k
       for j = 1:k
           count = 0;
           for m = 1:numRows
               if eq(y(m),i) && eq(Train_preds(m),j)
                  count = count+1; 
               end
           end
           confusion(i,j) =count;
       end
    end
     confusion
end

