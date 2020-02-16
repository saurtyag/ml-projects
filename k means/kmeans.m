%Author Saurabh Tyagi
%Loads the data from text file 
X = load('C:/Users/tysau/Downloads/SBU courses/ML/HW5/hw5data/digit/digit.txt');
y = load('C:/Users/tysau/Downloads/SBU courses/ML/HW5/hw5data/digit/labels.txt');
callQ251(X,y);
q253(X);
q254(X,y);

function [] = q254(X,y)
    random_vals = true;
    p_arr = zeros(10,3);
    for k=1:10
        [decision_clustering,~,~] = kmeans(X,k,random_vals);
        [~,p1,p2,p3,~,~,~] = findPairCountingMeasure(decision_clustering,y);
        p_arr(k,1) = p1;
        p_arr(k,2) = p2;
        p_arr(k,3) = p3;
    end
    
    figure
    y1 = p_arr(:,1);
    plot(y1);
    hold on
    
    y2 = p_arr(:,2);
    plot(y2);
    hold on
    
    y3 = p_arr(:,3);
    plot(y3);
    xlabel('k');
    ylabel('p1,p2,p3');
    legend('p1','p2','p3')
end

function [] = callQ251(X,y)
    k = 2;
    q251(X,y,k);
    k = 4;
    q251(X,y,k);
    k = 6;
    q251(X,y,k);
end

function [] = q253(X)
    random_vals = true;
    SOS_array = zeros(10,1);
    for k=1:10
        [decision_clustering,~,~] = kmeans(X,k,random_vals);
        SOS_array(k,1) =  findSumOfSquares(X,decision_clustering,k);
    end
    
    figure
    plot(SOS_array);
    xlabel('k');
    ylabel('Total within group sum of squares');
end

function [] = q251(X,y,k)
    random_vals = false;
    [decision_clustering,~,~] = kmeans(X,k,random_vals);
    [SOS] = findSumOfSquares(X,decision_clustering,k);
    [~,p1,p2,p3,~,~,~] = findPairCountingMeasure(decision_clustering,y);
    fprintf('k:%d SOS:%f p1:%f p2:%f p3:%f \n',k,SOS,p1,p2,p3);
end


function [decision_clustering,perm,count] = kmeans(X,k,random_vals)
    fprintf('New k means calculation starts\n');
    [numRows,~] = size(X); 
    count = 0;
    distances = zeros(numRows,k);
    decision_clustering = zeros(numRows,1);
    decision_clustering_old = zeros(numRows,1);
    if random_vals
        %case where centers are selected randomly
        perm = X(randperm(k),:);
    else
        %case where 1st k centers are selected as centers
        perm = X(1:k,:);
    end
    
    while count < 20
        %writing the loop for calculating the distances 
        for j=1:k
            input_RHS = X-perm(j,:);
            RHS = vecnorm(input_RHS,2,2);
            distances(:,j) = RHS;
        end
        [~,I] = min(distances,[],2);
        decision_clustering = I;
        
        if eq(decision_clustering_old,decision_clustering)
            fprintf('Count:%d  \n',count);
            return;
        else
            decision_clustering_old = decision_clustering;
        end

		for j=1:k
			curr_cluster_samples = X(decision_clustering == j,:);
            perm(j,:) = mean(curr_cluster_samples);
        end
        count = count + 1;
    end     
    fprintf('Count:%d  \n',count);
end

function [SOS] = findSumOfSquares(X,decision_clustering,k)
    
    cluster_means = [];
    total_ss = 0;
    for cluster_index = 1:k
        curr_cluster_samples = X(decision_clustering == cluster_index,:);
        curr_cluster_mean = mean(curr_cluster_samples);
        
        curr_ss = sum( sum( (curr_cluster_samples - curr_cluster_mean).^2,2 ));
        total_ss = total_ss + curr_ss;
       % cluster_means = [cluster_means;curr_cluster_mean];
    end
    
    SOS = total_ss;
end

%Here we will pass the calculated clustering and the actual label data and
%see how it fares, by finding the average of the same class points assigned
%to the same cluster and the differently classed points assigned to
%different clusters.
function [C,p1,p2,p3,count_same,count_diff,numRows] = findPairCountingMeasure(decision_clustering,y)
    %this will bring all the possible pairs into the fold and store it in X.
    y_index = [1:1000];
    C = nchoosek(y_index,2);
    [numRows,~] = size(C);
    count_same = 0;
    count_same_decision = 0;
    count_diff = 0;
    count_diff_decision = 0;
    for i=1:numRows
        if eq(y(C(i,1)),y(C(i,2)))
            count_same = count_same + 1;
            if eq(decision_clustering(C(i,1)),decision_clustering(C(i,2)))
                count_same_decision = count_same_decision +1;
            end
        end
        if ne(y(C(i,1)),y(C(i,2)))
            count_diff = count_diff + 1;
            if ne(decision_clustering(C(i,1)),decision_clustering(C(i,2)))
                count_diff_decision = count_diff_decision +1;
            end
        end
    end
    
    p1 = (count_same_decision/count_same)*100;
    p2 = (count_diff_decision/count_diff)*100;
    %p3 is the average of p1 and p2  
    p3 = (p1+p2)/2;
            
end
