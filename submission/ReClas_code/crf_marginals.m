function [P] = crf_marginals(X,trans,mu,sigma,init,semi)
    % [P] = crf_marginals(X,trans,mu,sigma,init,semi)
    %
    % X T by 1 observation sequence 
    % semi 0/1 if 1 make semi-Markov

    T=size(X,1); % length of observation (number of instances)
    C=4; % number of classes/states 
    %      1=normal 2=SIRS 3=severe sepsis 4=septic shock
    %initprobs = [0.5 0.45 0.05 0];
    
    % Compute P(y_t=c|x_t) for all c,t
    N = zeros(T,C);
    for i=1:C
        N(:,i) = normpdf(X,mu(i),sigma(i));
    end
    
    alpha = zeros(T,C);
    P = zeros(T,C);
    % Initialize forward algorithm with prior on P(y=c) and P(y1=c|x1)
    N(1,:) = N(1,:)/sum(N(1,:));
    alpha(1,:) = init.*N(1,:);
    P(1,:) = alpha(1,:)/sum(alpha(1,:));
    
    if semi
        dur = zeros(C+1,1);
    end
    
    for t=2:T
        % Normalize the probabilities of label | observation
        N(t,:) = N(t,:)/sum(N(t,:));
        % Forward algorithm
        alpha(t,:) = alpha(t-1,:)*trans'.*N(t,:);
        % Since we compute marginal at last time step, beta_t is always 1
        P(t,:) = alpha(t,:)/sum(alpha(t,:));
    end
    
end