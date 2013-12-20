[A, B, E] = textread('/Users/katie/Hopkins/PhysMod/ReClas/sepshock/csv_metadata_test','%f,%f,%f');
% icustay_id hospital_time true_label
[D] = textread('/Users/katie/Hopkins/PhysMod/ReClas/sepshock/decision_values_all','%f');
% probability of sepsis

ID = unique(A);

N = size(ID,1);
P = cell(N,1);
Time = cell(N,1);
Y = cell(N,1);
C=2;

% Separate into samples based on icutay_ID
for n=1:N
    inds = find(A==ID(n));
    P{n} = D(inds);
    Time{n} = B(inds);
    Y{n} = E(inds);
end

maxlen=max(cellfun(@(x) length(x), Y));
% 0 is no data
% 1 = -1 in Y is no septic shock
% 2 =  0 in Y is unknown (confounding event or 12h period before sepshock)
% 3 =  1 in Y is yes septic shock
Y_mat=zeros(maxlen,N);
% Estimate transition probabilities, classes are -1 , 1 
% (moved to 1,3 in Y_mat)
T = zeros(2);
dur_sep = zeros(N,1);
for n=1:N
    Y_mat(1:length(Y{n}),n) = Y{n} +2;
    tmp = arrayfun(@sigmoid,P{n});
    P{n} = [1-tmp tmp];
    OInds = find(Y{n} == -1);
    TInds = find(Y{n} == 1);
    ZInds = find(Y{n} == 0);
    dur_sep(n) = length(TInds);
    oo = intersect(OInds+1,OInds);
    ot = [intersect(OInds+1,TInds); intersect(ZInds+1,TInds)];
    to = intersect(TInds+1,OInds);
    tt = intersect(TInds+1,TInds);
%     if ~isempty(ot)
%         T(1,2) = T(1,2) + 1;
%     else
%         T(1,1) = T(1,1) + 1;
%     end
%     if ~isempty(to)
%         T(2,1) = T(2,1) + 1;
%     else
%         T(2,2) = T(2,2) + 1;
%     end
    obsv = size([OInds; TInds],1);
    if obsv>0
        T(1,1) = T(1,1) + length(oo)/obsv;
        T(1,2) = T(1,2) + length(ot)/obsv;
        T(2,1) = T(2,1) + length(to)/obsv;
        T(2,2) = T(2,2) + length(tt)/obsv;
    end
    
end
init = sum(T,1)/sum(sum(T));
% Use these as the weight for transitions
denom = sum(T,2);
T(1,1) = T(1,1)/denom(1);
T(1,2) = T(1,2)/denom(1);
T(2,1) = T(2,1)/denom(2);
T(2,2) = T(2,2)/denom(2); 


% For each sample chain into CRF and calculate marginals
Gamma = cell(N,1);
parfor n=1:N
    alpha = zeros(maxlen,C);
    M = zeros(maxlen,C);
    % Initialize forward algorithm with prior on P(y=c) and P(y1=c|x1)
    P_mat = P{n};
    alpha(1,:) = init.*P_mat(1,:);
    M(1,:) = alpha(1,:)/sum(alpha(1,:),2);
    if size(P_mat,1)<2
        continue;
    end
    for t=2:length(P_mat)
        % Forward algorithm
        alpha(t,:) = alpha(t-1,:)*T'.*P_mat(t,:);
        denom = sum(alpha(t,:),2);
        if denom > 0
            M(t,:) = alpha(t,:)/denom;
        else
            M(t,:) = alpha(t,:);
        end
    end
    Gamma{n} = M(1:length(P_mat),:);
end
    
S_a = [0 1; 1 0; 0.3 0.3];
[U_a,Up_a,~]=make_decisions(Gamma, Y_mat, ones(3,3),S_a);

S_b = [0 1; 0.75 0; 0.3 0.3];
[U_b,Up_b,~]=make_decisions(Gamma, Y_mat, ones(3,3),S_b);
    