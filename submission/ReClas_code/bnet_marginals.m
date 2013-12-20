function [P] = bnet_marginals(X,trans,mu,sigma)

    % X T by 1 observation sequence 

    T=size(X,1); % length of observation
    C=4; % number of classes 1=normal 2=SIRS 3=severe sepsis 4=septic shock
    initprobs = [0.5 0.45 0.05 0];
    
    Sigma = zeros(C,C,C);
    for i=1:4
        Sigma(1,1,i) = sigma(i)*eye(1,1);
    end

    intra = zeros(2);
    intra(1,2) = 1; 
    inter = zeros(2);
    inter(1,1) = 1;
    node_sizes = [4 1];
    hnodes=1;
    onodes=2;
    eclass1 = [1 2];
    eclass2 = [3 2];
    eclass = [eclass1 eclass2];
    
    
    bnet=mk_dbn(intra,inter, node_sizes, 'discrete', hnodes,'eclass1', eclass1,'eclass2',eclass2, 'observed', onodes);
    bnet.CPD{1} = tabular_CPD(bnet, 1,'CPT', initprobs); % priors
    bnet.CPD{3} = tabular_CPD(bnet, 3, 'CPT', trans);
    bnet.CPD{2} = gaussian_CPD(bnet, 2, 'mean', mu, 'cov', sigma);
    

    
     
%     evidence(T+1:T+T) = num2cell(X); 
    P = zeros(T,C);
    
    parfor t=1:T
        % enter observation x_t
        % do this incrementally so at each iteration we only get marginals
        % based on observed data rather than influence from downstream
        % (i.e. future) data
        engine = smoother_engine(hmm_2TBN_inf_engine(bnet));
        evidence = cell(2, length(X));
        evidence(2,1:t) = num2cell(X(1:t));
        [engine, loglik] = enter_evidence(engine, evidence);
        marg = marginal_nodes(engine,1,t);
        P(t,:)=marg.T';
    end
    
end