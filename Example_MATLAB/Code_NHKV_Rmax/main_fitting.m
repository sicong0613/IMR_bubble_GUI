
clear; clc; close all;

%% Loading data

%%%%% Far-field thermodynamic conditions %%%%%
Pinf = 101325; % Atmospheric pressure (Pa)
T_inf = 298.15; % Far field temperature (K)
rho = 1000;
c_long = 1486; %  1485;  % Longitudinal wave speed (m/s)
gamma = 0.0725; % Surface tension (N/m)

px2um = 3.2; % 3.2 um per pixel
fps = 1e6; % frames per second

%cav_type = input('Please fill in the type of cavitation driving force (LIC or UIC): ', 's');
%cav_type = 'LIC';



%% read data
% Open a file selection dialog box
BoxFolderPath = ['/Users/yujiezhang/Library/CloudStorage/Box-Box/Yujie_Version_Management/ResearchProjects_Yujie'];
path_bubble_Rt = [BoxFolderPath, '/LIC/001_LIC_Gradient'];

% BoxFolderPath = 'C:\Users\jy24967\Box\research_projects\Weekly_individual_report\Individual_folder_Yujie_Zhang';
% path_bubble_Rt = [BoxFolderPath, '\code\multiplebubble_Yujie\Jin_15_50_40'];

[t_exp_shift, t_shift_nondim, t_exp, t_nondim, tc, ...
    R_nondim, Rmax_um, Req_m, R_m, fileName] = fun_read_data(px2um, fps, Pinf, rho, path_bubble_Rt);

% Req_exp: m; R_m: bubble radius, in unim m (meter).


plot(t_exp_shift, R_m)
tspan2fit_exp = t_exp(end);

% R = R_m;
% t = t_exp_shift;
% 
% path_save_Rt =  [BoxFolderPath, '/LIC/001_LIC_Gradient/GUI_fitting_Rt/20260406'];
% fullSavePath = fullfile(path_save_Rt, [fileName, '.mat']);
% save(fullSavePath, 'R', 't');
% % fullSavePath_para = fullfile(path_save_Rt, [fileName, '_par', '.mat']);
% % save(fullSavePath_para, 'Req_m', 'Pinf', 'rho', 'gamma', 'c_long', 'T_inf');

%%

% find the time used for fitting
%  [t_win_nondim, R_exp_win_nondim, t_end_nondim] = Find_t_and_R(R_nondim, t_nondim); 
% tspan_nondim = t_end_nondim;
% figure
% plot(t_win_nondim, R_exp_win_nondim)
[~, index_start] = max(R_nondim);

tspan_nondim = t_nondim(end);

% Fit from the maximum radius to the first collapse minimum.
% Avoid fixed end indices, which can miss the collapse on some datasets.
R_after_max = R_nondim(index_start:end);
local_min_rel = find( ...
    R_after_max(2:end-1) <= R_after_max(1:end-2) & ...
    R_after_max(2:end-1) <  R_after_max(3:end) & ...
    R_after_max(2:end-1) <= 0.95 * R_after_max(1), ...
    1, 'first') + 1;

if isempty(local_min_rel)
    [~, local_min_rel] = min(R_after_max);
end

index_end = index_start + local_min_rel - 1;

t_win_nondim = t_nondim(index_start:index_end);
R_exp_win_nondim = R_nondim(index_start:index_end);

tspan_s = t_exp(end);
t_win_s = t_exp(index_start:index_end);
R_exp_win_m = R_m(index_start:index_end);

[~, idx_exp_max] = max(R_exp_win_m);
t_exp_max = t_win_s(idx_exp_max);
t_win_s = t_win_s - t_exp_max;

%% Fit U, G and mu (using primary bubble)
 %U0p = 67; % initial speed, to fit
 Gp = 1.2e4;     % Ground-state shear modulus (Pa) %%%%% TODO
 mup = 0.13;     % Viscosity (Pa-s) %%%%% TODO

 var2fit0 = [Gp,mup];

%U0p_lb = 40;     U0p_ub = 120;
G_lb = 1;      G_ub = 1e6;
mu_lb = 0.01;  mu_ub = 1;
lb_primary = [G_lb, mu_lb];
ub_primary = [G_ub, mu_ub];

% options
options = optimoptions('patternsearch',...
        'Display','iter', ...
        'Algorithm','classic', ...          % clearer “Successful/Unsuccessful Poll” logging
        'UseCompletePoll',true, ...
        'PollMethod','GSSPositiveBasis2N', ...
        'ScaleMesh', true, ...              % you already log-transform
        'InitialMeshSize',1.0, ...
        'MeshContractionFactor',0.8, ...
        'MeshTolerance',1e-10, ...
         'StepTolerance',1e-10, ...
        'Cache','on','CacheTol',1e-12, ...
        'UseParallel', true);


[var2fit_best_primary, fval, exitflag, output] = patternsearch( @(var2fit)...
    fun_LSQErr_primary_Bubble2(var2fit, ...
    t_exp, tspan_nondim, t_nondim, ...
    Req_m, R_nondim, Rmax_um, ...
    t_win_s, R_exp_win_m),...
    var2fit0, [], [], [], [], lb_primary, ub_primary, [], options);


%%
var_best_fit = var2fit_best_primary;
tspan_nondim_2 = tspan2fit_exp/tc;
[t_sim, t_sim_nondim, R_sim_nondim] = ...
fun_Bubbles_fitting_results(var_best_fit, ...
    t_nondim, tspan_nondim_2, ...
    Req_m, Rmax_um);

[Rsim_max, Index_Rsim_max] = max(R_sim_nondim);
time_sim_shift = t_sim - t_sim(Index_Rsim_max);
time_sim_shift_nondim = t_sim_nondim - t_sim_nondim(Index_Rsim_max);

figure
plot(time_sim_shift_nondim, R_sim_nondim,'LineWidth',2)
hold on 
scatter(t_shift_nondim, R_nondim, 30, 'filled')
%plot(t_shift_nondim, R_nondim,'*', 'MarkerSize', 8)
legend('Sim', 'Exp', 'Fontsize', 20)

xlabel('time (nondim)','FontSize',20)
ylabel('R/R_{max}','FontSize',20)

ax = gca;
ax.FontSize = 20;
xlim([0,2])
ylim([0,1])




%%
var = [6.6945e3 0.1390];

tspan_nondim_2 = tspan2fit_exp/tc;
[t_sim, t_sim_nondim, R_sim_nondim] = ...
fun_Bubbles_fitting_results(var, ...
    t_nondim, tspan_nondim_2, ...
    Req_m, Rmax_um);

[Rsim_max, Index_Rsim_max] = max(R_sim_nondim);
time_sim_shift_nondim = t_sim_nondim - t_sim_nondim(Index_Rsim_max);

figure
plot(time_sim_shift_nondim, R_sim_nondim,'LineWidth',2)
hold on 
scatter(t_shift_nondim, R_nondim, 30, 'filled')
%plot(t_exp_shift, R_nondim, 'LineWidth',2)
legend('Sim', 'Exp', 'Fontsize', 20)

xlabel('time (s)','FontSize',20)
ylabel('R/R_{eq}','FontSize',20)

ax = gca;
ax.FontSize = 20;

%xlim([0, 1.5])
ylim([0,1])

%%
% var_best_fit = [1e5 0.1 40];
% tspan2fit_exp =  tspan2fit_exp;
% [t_sim,R_sim,U_sim,P_sim, t_sim_nondim,R_sim_nondim] = ...
% fun_IMR_damage(var_best_fit, t_nondim, R_nondim, Rmax_um, Req_exp, tspan2fit_exp);
% figure
% plot(t_sim, R_sim)
% 
