%does an XOR task

% %number of inputs
N = 2;

%XOR
%%%%%%%%%%%%%%%%%%%%%%
T = 500;
[x1,x2,target] = generate2bitparity(T);
X = [x1,x2]';
original_target = target;
%%%%%%%%%%%%%%%%%%%%%%

%learning rate
eta = 0.2; %default: 0.2
sigma = 0.6;
tau = 0.001;
% tau_b = 0.001;

%number of recurrent units
M = 1500;

%randomized initial conditions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W_in = randn(M,N);
W = randn(M);
W_out = randn(1,M);
B = randn(N,M);
previous_K = randn(M,T);
Yhat = randn(T,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%activate the recurrent network
I = W_in*X; %input
u = I + W*previous_K; %total activation
K = tanh(u); %default

save_pc_err = zeros(num_train_iter,1);
save_err = zeros(num_train_iter,1);
save_errY = zeros(num_train_iter,1);
save_W_out = zeros(num_train_iter,M);
save_Ystar = zeros(num_train_iter,M);
class_err = zeros(num_train_iter,1);
for iter = 1:num_train_iter
    
    %output
    Ystar = K'*W_out';
    
    class_err(iter) = getErr(Ystar,target);
    
    %train output weights
    err = Ystar - target;
    errY = Ystar - Yhat;
    
    %default rule: (i.e., truncated rule)
    dW_out = -eta*K*err - sigma*K*errY;
    W_out = W_out+tau.*dW_out';
    
    %update prediction
    dYhat = -Yhat + Ystar;
    Yhat = Yhat + tau_b*dYhat;
    
    err = Ystar - original_target;
    save_err(iter) = nanmean(err.^2);
    
    save_errY(iter) = nanmean(errY.^2); %prediction error
end

