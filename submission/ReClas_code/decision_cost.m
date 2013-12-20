function C=decision_cost(U,Y,S)
    % C=decision_cost(U,Y,S)
    % C 1x1 cost value
    % U Nx1 decision policy
    % Y Nx1 true labels
    % S C+1xC cost matrix
    C= zeros(size(U));
    C(1) = S(U(1),Y(1));
    for n=2:size(U,1)
        C(n)=C(n-1) + S(U(n),Y(n));
    end
    
end