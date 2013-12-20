function [P] = bnet_marginals(X)

    % X T by 1 observation sequence       

    T=size(X,1); % length of observation
    C=4; % number of classes 1=normal 2=SIRS 3=severe sepsis 4=septic shock
    initprobs = [0.5 0.45 0.05 0];
    
    % Assume transition matrix and means of Gaussians are given
    trans = [0.95 0.05 0 0; 0.01 0.95 0.04 0; 0 0.02 0.95 0.03; 0 0 0.05 0.95];
    mu = [-2 0 1 2]; % mu of Gaussian for each class
    sigma = [5 2 0.5 0.1]; % sigma of Gaussian for each class

    % Create Bayes Net
    dag=eye(T,T);
    dag=[zeros(T,1) dag(1:T,1:T-1) zeros(T); dag zeros(T)];
    node_sizes=C*ones(1,T);
    bnet=mk_bnet(dag, node_sizes, 'discrete', 1:T, 'observed', T+1:T+T);
    bnet.CPD{1} = tabular_CPD(bnet, 1, initprobs);
    bnet.CPD{T+1} = gaussian_CPD(bnet, T+1, 'mean', mu, 'cov', sigma);
    
    for t=2:T        
        bnet.CPD{t} = tabular_CPD(bnet, t, 'CPT', trans);
        bnet.CPD{t+T} = gaussian_CPD(bnet, t+T, 'mean', mu, 'cov', sigma);
    end
    engine = jtree_inf_engine(bnet);
    evidence = cell(1, T+T);
    evidence(T+1:T+T) = num2cell(X); 
    [engine, loglik] = enter_evidence(engine, evidence);
    P = zeros(T,C);
    
    for t=1:T
        marg = marginal_nodes(engine,t);
        P=marg.T;
        % Add in P(U_t|U_t-1)?
        E = S*P;
        V=V+E;
        U(t,1) = argmin(V); % will choose first element if two are equal
    end
    
end