% Generate synthetic data

T=250; % length of observation
C=4; % number of classes 1=normal 2=SIRS 3=severe sepsis 4=septic shock
N = 1000; % number of samples to generate
initprobs = [0.5 0.45 0.05 0];
alpha=0.25;
S = [0 1 1 1; 1 0 1 1; 1 1 0 1; 1 1 1 0; alpha alpha alpha alpha];

trans = [0.98 0.02 0 0; 0.01 0.97 0.02 0; 0 0.01 0.97 0.02; 0 0 0.02 0.98];
mu = [0 4 8 10]; % mu of Gaussian for each class
sigma = [3 3 2 1]; % sigma of Gaussian for each class

% generate synthetic data using intial probabilities (initporbs) and
% transition matrix (trans)
% X Txns matrix of observations
% Y Txns matrix of class labels
[X,Y] = synthetic_data(T,C,N,initprobs,trans,mu,sigma,0.1);

% Get marginal probabilities for each observation sequence
% Calculate the expected cost at that time point based on the prior time


P_hmm = cell(N,1);
P_crf = cell(N,1);
U = zeros(T,N); % decisions allowing for rejection
UP = zeros(T,N); % decisions without allowing rejection
V = cell(N,1); % value with allowing for rejection
A = zeros(C+2,C+1); % counts for accuracy score w/reject
AP = zeros(C+2,C+1); % counts w/o reject

% P(u_t|u_t-1)
TM = ones(C+1,C+1);
% Focus on rest and add trans probs later for additional smoothing
%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%
%TM(1:C,1:C) = trans;
%%%%%%%%%%%%%%%%%%%%%%


for n=1:N
    fprintf('Computing marginals: sample %d\n', n);
    
    % compute marginal probabilities
    % with HMM
    P_hmm{n} = bnet_marginals(X(:,n),trans,mu,sigma);
    % with CRF
    P_crf{n} = crf_marginals(X(:,n),trans,mu,sigma,initprobs);

    % Run decision process for each P and get sequence of decisions
    % Find better way of doing this will cellfun
    [u,v,up] = reclas_decisions(P_crf{n}, S, TM);
    U(:,n) = u';
    UP(:,n) = up';
    A = A + decision_accuracy(u',Y(:,n));
    AP = AP + decision_accuracy(up',Y(:,n));
    
    V{n} = v;

end

L = A(1:C+1,1:C).*S;
LP = AP(1:C,1:C).*S(1:C,:);

L = [L sum(L,2)];
L = [L; sum(L,1)];

LP = [LP sum(LP,2)];
LP = [LP; sum(LP,1)];
total = A(C+2,C+1); % total number of samples
acc = sum(diag(A(1:C,1:C))); % number accurately classified
ref = A(C+1,C+1);

stats = zeros(1,8);
misInds = find(UP ~= Y);% indices where DP w/o refusal makes mistake
corrInds = find(UP == Y);
ctotal = size(corrInds,1); % total number correctly classified
mtotal = total - ctotal; % total number misclassified
stats(1,1) = L(C+2,C+1)/total; % loss
stats(1,2) = acc/total; % accuracy
stats(1,3) = (total -acc - ref)/total; % error
stats(1,4) = ref/total; % refusal rate
stats(1,5) = size(find(U(corrInds) == C+1),1)/ctotal; % corr -> ref
stats(1,6) = size(find(U(corrInds) ~= C+1),1)/ctotal; % corr -> corr
stats(1,7) = size(find(U(misInds) == C+1),1)/mtotal; % mis -> ref
stats(1,8) = size(find(U(misInds) ~= C+1),1)/mtotal; % mis -> mis

L
LP