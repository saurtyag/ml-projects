% author Saurabh Tyagi

E_All = zeros(30,1);
V_All = zeros(30,1);
CV_All = zeros(30,1);
length_func_call = 30;
%for question 2.1
N = 10;
for k = 1:30
[E_All(k),V_All(k),CV_All(k)]=question2(N);
end
mean_E = mean(E_All)
sd_E = sqrt(sum((E_All-mean_E).^2)/(length_func_call-1))
mean_V = mean(V_All)
sd_V = sqrt(sum((V_All-mean_V).^2)/(length_func_call-1))
mean_C = mean(CV_All)
sd_C = sqrt(sum((CV_All-mean_C).^2)/(length_func_call-1))


%for question 2.2
N = 100;
for k = 1:30
[E_All(k),V_All(k),CV_All(k)]=question2(N);
end
mean_E2 = mean(E_All)
sd_E2 = sqrt(sum((E_All-mean_E2).^2)/(length_func_call-1))
mean_V2 = mean(V_All)
sd_V2 = sqrt(sum((V_All-mean_V2).^2)/(length_func_call-1))
mean_C2 = mean(CV_All)
sd_C2 = sqrt(sum((CV_All-mean_C2).^2)/(length_func_call-1))

function [ E,V,C ] = question2( N )
%question2 Returns the Expectation, Variance and Covariance scalars
%   The function returns the Expectation, Variance and Covariance as
%   scalars. The established information here is that E(X),V(X),C(X,X1)
%   will be calculated for X = max(X1,X2) and X1, X2 are uniform random
%   variables over the support range [0,1].

X1 = zeros(N,1);
X2 = zeros(N,1);
% here we are creating the input to our max function, and vectors for X1 and X2
for i = 1:N
    X1(i) = rand;
    X2(i) = rand;
end
X = max(X1,X2); 

%The following line of code will calculate the mean based on the input N
%provided by the user.
E = mean(X);

%For variance, we follow the derived formula E(X^2)-[E(X)]^2

%calculation of Variance
Xvar = X.^2;

V = (mean(Xvar) - (E.^2));


%For covariance, the derived formula is E[(X-E(X))(Y-E(Y))] which 
% simplifies to E[XY]-E[X]E[Y]. 

% calculation of Covariance
val1 = mean(X.*X1);
val2 = mean(X)*mean(X1);
C = val1 - val2;

end