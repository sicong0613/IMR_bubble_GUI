% clear; clc; close all;

% mat_to_transfer = "simulation_result\DNA_0530_160946.mat";
% load(mat_to_transfer);

% R = R_sim;
% t = t_sim;
um2px = 3.2;
fps = 2e6;
R = [ a_lengths((a_lengths > 0))]*um2px*1e-6 ;
t = [ 1 : 1 : length(R)]/fps ; % Exp raw time points (unit:  s)

% Pinf = 101325; % unit: Pa
% rho = 1000; % unit: kg/m^3

R_eq =mean(R(end-20:end));

save("test.mat", "R", "t", "R_eq");