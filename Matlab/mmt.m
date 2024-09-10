function obj = mmt(x,N,T,Psi,Sig,I_pi,I_gdp,I_y1,I_yspr,I_cy, ...
    inflpos,gdppos,y1pos,ysprpos,pi0,x0,ynom1q0,yspr0,cy0,X2,yielddata,yieldmaturity,eps2)

striphorizon = 1e2;

L0 = zeros(N,1);
L1 = zeros(N,N);

L0(1:4) = x(1:4)';
tmp = zeros(4:4);
tmp(:) = x((4+1):(4+4^2));

L1(1:4,1:4) = tmp./std(X2(:,1:4));





FF = [];

if (all(isnan(x)))
    obj = 1e20;
    return;
end


Lt = L0+L1*X2';


mean_sr = mean(sqrt(diag(Lt'*Lt)));
iota = 0.001;
soft_penalty_param = 1e4;
FF = [FF ,soft_penalty_param * 1/(1 + exp(-(mean_sr - 0.36)/iota))];
%if (mean_sr > 0.36 + 1e-8)
%    obj = 1e20 * (1 + (mean_sr - 0.35));
%    return;
%end

Bpibar  = (-I_y1'*inv(eye(N) - (Psi-Sig*L1)))';

dApibar = - ynom1q0 + .5*Bpibar'*(Sig*Sig')*Bpibar - Bpibar'*Sig*L0;

iota = 0.0001;
FF = [FF, soft_penalty_param * 1 / (1 + exp(- ((0.005 - 4*(-dApibar)) / iota)))];

%if (-dApibar < 0.25/100 - 2e-8) % at least 1% nominal rate per annum in super long run
%    obj = 1e20 * (1 + abs(-dApibar - 0.25/100));
%    return;
%end

% %% arbitrage-free definition of real risk-free rate
% y0_1   =  ynom1q0 - pi0 - .5*I_pi'*(Sig*Sig')*I_pi + I_pi'*Sig*L0; 

%% No arbitrage restrictions on bond yields
Api     = zeros(striphorizon,1); 
Bpi     = zeros(N,striphorizon);

Api(1)  = -ynom1q0 + cy0; 
Bpi(:,1)= -I_y1' + I_cy';

for j = 1:striphorizon
    Api(j+1)  = - ynom1q0 + Api(j) + .5*Bpi(:,j)'*(Sig*Sig')*Bpi(:,j) - Bpi(:,j)'*Sig*L0;
    Bpi(:,j+1)= (Bpi(:,j)'*Psi- I_y1'- Bpi(:,j)'*Sig*L1)'; 
end

iota = 0.001;
FF = [FF , soft_penalty_param * 1 / (1 + exp(-((Api(end) - 0) / iota)))];
%if (Api(end)>0) 
%    obj = 1e20;
%    return;
%end

%% Insist on matching the 5-year yield more closely since it is in the state space
format long

penalty = 1e8;
FF_newa    = (100*(-(Api(20)/20) - (ynom1q0+yspr0))).^2*penalty;


FF         = [FF FF_newa];
FFtmp     = (10*(-(Bpi(:,20)'/20) - (I_y1'+I_yspr'))).^2*penalty;
FF        = [FF, FFtmp];

%% Match Yield Curve
% Pricing nominal bond yields of maturities stored in yieldmaturity

predicted_yield = kron(ones(T,1),-Api(yieldmaturity)'./yieldmaturity) - ((Bpi(:,yieldmaturity)'./kron(yieldmaturity',ones(1,N)))*X2')';

Nom_error = 100*(predicted_yield - yielddata);
penalty = 1e4;
FF_new    = penalty*mean([Nom_error.^2]);
FF        = [FF FF_new];

obj = nansum(FF);


end

