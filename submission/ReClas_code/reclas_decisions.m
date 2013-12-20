function [U,Vh,UP] = reclas_decisions(P,S,TM)

% P TxC matrix P_t,c marginal likelihood of class c at time t
% S C+1xC cost matrix S_i,j cost of substituting i for j
% TM C+1xC+1 matrix of probability of P(U_t|U_t-1)

T = size(P,1); % number of observations in sequence
C = size(P,2); % number of classes, class C+1 is refuse to predict

U = zeros(T,1); % decision policy over sequence
UP =  zeros(T,1); % decision policy without rejection
V = zeros(C+1,1); % current cost values
VP = zeros(C,1); % current cost values
beta = 0.9; % discount factor to apply to past observations

Vh = zeros(T,C+1); % history of values that lead to decision
decprob = ones(1,C+1);
decprobP = ones(1,C+1);

for t=1:T
    % Compute the expected cost at time t
    if t>1
        decprob = TM(U(t-1),:);
        decprobP = TM(UP(t-1),:);
    end
    V(1:C)= beta^(T-t)*min(V) + S(1:C,:)*P(t,1:C)'.*decprob(1,1:C)';
    V(C+1) = beta^(T-t)*min(V) + S(C+1,:)*P(t,1:C)'.*decprob(1,C+1);
    U(t) = argmin(V);
    
    % Compute the expected cost at time t without rejection
    VP = beta^(T-t)*min(VP) + S(1:C,:)*P(t,:)'.*decprobP(1:C)';
    UP(t) = argmin(VP);
    
    Vh(t,:) = V';
end


end