function x= class_sample(c)

    mu = [0 3 6 8]; % mu of Gaussian for each class
    sigma = [3 3 2 1]; % sigma of Gaussian for each class
    x = normrnd(mu(c),sigma(c),[1 1]);

end

