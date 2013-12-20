function [D, labels] = synthetic_data(T,C,ns,initprobs,trans,mu,sigma,weight)

% T=100; % length of observation
% C=4; % number of classes 1=normal 2=SIRS 3=severe sepsis 4=septic shock
% ns = 1000; % number of samples to generate
% initprobs = [0.5 0.45 0.05 0];
% trans CxC transition matrix
% trans = [0.95 0.05 0 0; 0.01 0.95 0.04 0; 0 0.02 0.95 0.03; 0 0 0.05 0.95];
% weight double in [0,1] that specifies how much weight to give to
%    new sampled value 

% label is class from which new value is sampled (regardless of weight)
labels = zeros(T,C); 
D = zeros(T,ns);
for n=1:ns
    tic
    
    % Generate the first sample from the initial probabilities
    c = randsample(C,1,1,initprobs);
    labels(1,n) = c;
    D(1,n) = normrnd(mu(c),sigma(c),[1 1]);
    for t=2:T
        % Generate subsequent samples from transition matrix
        c = randsample(C,1,1,trans(labels(t-1,n),:)); % class to draw sample from
        labels(t,n) = c;
        % Weight influence of new sample by weight
        S = normrnd(mu(c),sigma(c),[1 1]);
        D(t,n) = D(t-1,n)*(1-weight) + S*weight;
    end
    fprintf('Generated sample %d. ', n);
    toc
end

%D = arrayfun(@class_sample_vec, labels);

end



