function [N,T,Psi,Sig,I_pi,...
          I_gdp,I_y1,I_yspr,I_cy,inflpos,gdppos,y1pos,...
          ysprpos,pi0,x0,ynom1q0,...
          yspr0,cy0,X2,tmpyields,yieldmaturity,eps2] = ...
        setup_model(tmpyields, yieldmaturity, cy, ...
                               numsearch, param_start, infl, x, ...
                               optionssimplex, optionsderiv)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Assembly
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % define the ordering in the VAR
    inflpos    = 1;
    gdppos     = 2;
    y1pos      = 3;
    ysprpos    = 4;
    cypos      = 5;

    ynom1q    = tmpyields(:,1)+cy;
    yspr      = tmpyields(:,6)-ynom1q; % spread between 5-yr yield and 3-month
    pi0       = mean(infl);
    x0        = mean(x);
    ynom1q0   = mean(ynom1q);
    yspr0     = mean(yspr);
    cy0       = mean(cy);

    X2(:,inflpos)  = infl;
    X2(:,gdppos)   = x;
    X2(:,y1pos)    = ynom1q;
    X2(:,ysprpos)  = yspr;
    X2(:,cypos)    = cy;

    X2 = X2 - mean(X2);

    T         = size(X2,1);
    N         = size(X2,2);
    I         = eye(N);
    I_pi      = I(:,inflpos);
    I_gdp     = I(:,gdppos);
    I_y1      = I(:,y1pos);
    I_yspr    = I(:,ysprpos);
    I_cy      = I(:,cypos);

    Y         = X2(2:end,:);
    X         = X2(1:end-1,:);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Estimate Psi and Sig
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Psi = zeros(N,N);
    Psi_se = zeros(N,N);

    for i = [inflpos gdppos y1pos ysprpos]
        regr         = ols(Y(:,i),[ones(T-1,1),X(:,1:4)]);
        c(i)         = regr.beta(1);
        c_se(i)      = regr.bstd(1);
        Psi(i,1:4)     = regr.beta(2:end)';
        Psi_se(i,1:4)  = regr.bstd(2:end)';
        R2(i)        = regr.rsqr;
        R2bar(i)     = regr.rbar;
        eps(:,i)     = Y(:,i) - c(i) - X*Psi(i,:)';
        %     tstat(:,i)   = regr.tstat(2:end)';
        clear regr;
    end
    
    if (std(Y(:,cypos)) == 0)
        Y(:,cypos) = Y(:,cypos)*0;
        Psi(cypos,:)   = 0;
%         Psi(cypos,cypos) = 1;
        eps(:,cypos)   = 0;
        Sigma = cov(eps(:,1:4));
        Sig   = zeros(N,N);
        Sig(1:4,1:4)   = chol(Sigma,'lower');
    else 
        for i = [cypos]
            regr         = ols(Y(:,i),[ones(T-1,1),X(:,:)]);
            c(i)         = regr.beta(1);
            c_se(i)      = regr.bstd(1);
            Psi(i,:)     = regr.beta(2:end)';
            Psi_se(i,:)  = regr.bstd(2:end)';
            R2(i)        = regr.rsqr;
            R2bar(i)     = regr.rbar;
            eps(:,i)     = Y(:,i) - c(i) - X*Psi(i,:)';
            %     tstat(:,i)   = regr.tstat(2:end)';
            clear regr;
        end
        Sigma = cov(eps);
        Sig   = chol(Sigma,'lower');
    end

%     max(abs(eig(Psi)));
    eps2 = eps;
    
end

