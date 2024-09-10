%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('../CleanData/debt.mat')

date       = 1971.5:.25:2020.75; % Q1 is .00 and Q4 is .75, 1971Q3-2020Q4

m = m/4; % quarterly interest rates

ynom1q      = m(:,1); 
ynom1y      = m(:,2); 
ynom2y      = m(:,3); 
ynom3y      = m(:,4); 
ynom4y      = m(:,5); 
ynom5y      = m(:,6); 
ynom7y      = m(:,8); 
ynom10y      = m(:,11); 
ynom15y      = m(:,16); 
ynom20y      = m(:,21); 
ynom30y      = m(:,31); 

yielddata = [ynom1q,ynom1y,ynom2y,ynom3y,ynom4y,ynom5y,ynom7y,...
             ynom10y];
yieldmaturity = [1,4,8,12,16,20,28,40];

load('../CleanData/macro.mat')
% m = xlsread('../CleanData/macro.csv'); 

infl       = m(:,1);  % growth in GDP price deflator (in logs)
x          = m(:,2);  % real gdp growth (in logs)


load('../CleanData/convyield.mat')
m = m/4; % quarterly 

cy_diff       = m(:,1) - ynom1q;  % conv yield from 3m CD minus 3m Tbill
