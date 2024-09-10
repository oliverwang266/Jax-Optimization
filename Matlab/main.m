clear all; close all; clc
loaddata;
% number of params
nparam = 20;

% values of convenience yields
allcy = 0:0.05:2.5;
cylen = length(allcy);

load('results.mat')

maxval = size(minparamval, 2);

idx = 1;
cy = allcy(idx) * cy_diff;
%param = readmatrix('param1.csv');
param = minparamval(:, idx);

[N, T, Psi, Sig, I_pi, I_gdp, I_y1, I_yspr, I_cy, inflpos, gdppos, y1pos, ...
     ysprpos, pi0, x0, ynom1q0, ...
     yspr0, cy0, X2, yielddata, yieldmaturity, eps2] = ...
    setup_model(yielddata, yieldmaturity, cy, ...
    [], [], infl, x, ...
    [], []);

mmt(param, N, T, Psi, Sig, I_pi, I_gdp, I_y1, I_yspr, I_cy, ...
    inflpos, gdppos, y1pos, ysprpos, pi0, x0, ynom1q0, yspr0, cy0, X2, yielddata, yieldmaturity, eps2)

% optimizer
tic
param = param * 0;
param(1:4) = [0.25, 0.25, 0.25, 0.25];

optionssimplex = optimset('TolX', 1e-16, 'TolFun', 1e-16, 'MaxIter', 1e5, ...
    'MaxFunEval', 1e8, 'display', 'iter');
for ii = 1:10
    [param, fval] = ...
    fminsearch('mmt', param, optionssimplex, N, T, Psi, Sig, I_pi, ...
    I_gdp, I_y1, I_yspr, I_cy, inflpos, gdppos, y1pos, ...
    ysprpos, pi0, x0, ynom1q0, ...
    yspr0, cy0, X2, yielddata, yieldmaturity, eps2);
end
toc

mmt(param, N, T, Psi, Sig, I_pi, I_gdp, I_y1, I_yspr, I_cy, ...
    inflpos, gdppos, y1pos, ysprpos, pi0, x0, ynom1q0, yspr0, cy0, X2, yielddata, yieldmaturity, eps2)

% % plot 
% L0 = zeros(N, 1);
% L1 = zeros(N, N);

% L0(1:4) = param(1:4)';
% tmp = zeros(4:4);
% tmp(:) = param((4 + 1):(4 + 4 ^ 2));
% L1(1:4, 1:4) = tmp ./ std(X2(:, 1:4));

% striphorizon = 10000;
% Api = zeros(striphorizon, 1);
% Bpi = zeros(N, striphorizon);

% Api(1) = -ynom1q0 + cy0;
% Bpi(:, 1) = -I_y1' + I_cy';

% for j = 1:striphorizon
%     Api(j + 1) =- ynom1q0 + Api(j) + .5 * Bpi(:, j)' * (Sig * Sig') * Bpi(:, j) - Bpi(:, j)' * Sig * L0;
%     Bpi(:, j + 1) = (Bpi(:, j)' * Psi - I_y1' - Bpi(:, j)' * Sig * L1)';
% end

% Bpibar = (-I_y1' * inv(eye(N) - (Psi - Sig * L1)))';
% y_inf(idx) =- (- ynom1q0 + .5 * Bpibar' * (Sig * Sig') * Bpibar - Bpibar' * Sig * L0);
% y_30y(idx) = -Api(30) / 30;
% Bpi4(:, idx) = Bpi(:, 4);

% score0(idx) = mmt(param, N, T, Psi, Sig, I_pi, ...
%     I_gdp, I_y1, I_yspr, I_cy, inflpos, gdppos, y1pos, ...
%     ysprpos, pi0, x0, ynom1q0, ...
%     yspr0, cy0, X2, yielddata, yieldmaturity, eps2);
% Api_end(idx) = Api(end);

% predicted_yield = kron(ones(T, 1), -Api(yieldmaturity)' ./ yieldmaturity) - ((Bpi(:, yieldmaturity)' ./ kron(yieldmaturity', ones(1, N))) * X2')';
% Nom_error = 100 * (predicted_yield - yielddata);

% penalty = 1e4;
% score0_no5yrestriction(idx) = penalty * sum(mean([Nom_error .^ 2]));

% score(idx) = sqrt(mean(mean([Nom_error .^ 2]))) * 4;
% score_full(idx, :) = sqrt(mean(Nom_error .^ 2)) * 4;
% score_full_meanonly(idx, 1) = sqrt((100 * (- (Api(20) / 20) - (ynom1q0 + yspr0))) .^ 2);
% score_full_meanonly(idx, 2:6) = sqrt((100 * (- (Bpi(:, 20)' / 20) - (I_y1' + I_yspr'))) .^ 2);



% plot(1:T, predicted_yield(:, end), 1:T, yielddata(:, end))

% % plot(1:T, predicted_yield(:, 2), 1:T, yielddata(:, 2))
