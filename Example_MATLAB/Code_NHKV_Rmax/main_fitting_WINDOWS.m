%%

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
% BoxFolderPath = 'C:\Users\YZ29794\Box\Yujie_Version_Management/ResearchProjects_Yujie';
% path_bubble_Rt = [BoxFolderPath, 'ResearchProjects_Yujie/LIC_Gradient/CuringTime_Location'];
path_bubble_Rt = 'data';

[t_exp_shift, t_shift_nondim, t_exp, t_nondim, tc, ...
    R_nondim, Rmax_um, Req_m, R_m, fileName] = fun_read_data(px2um, fps, Pinf, rho, path_bubble_Rt);

% Req_exp: m; R_m: bubble radius, in unim m (meter).


plot(t_exp_shift, R_m)
tspan2fit_exp = t_exp(end);

% R = R_m;
% t = t_exp_shift;
%
% path_save_Rt =  [BoxFolderPath, '\LIC\001_LIC_Gradient\GUI_fitting_Rt'];
% fullSavePath = fullfile(path_save_Rt, [fileName, '.mat']);
% save(fullSavePath, 'R', 't');
% fullSavePath_para = fullfile(path_save_Rt, [fileName, '_par', '.mat']);
%save(fullSavePath_para, 'Req_m', 'Pinf', 'rho', 'gamma', 'c_long', 'T_inf');

%%

% find the time used for fitting
%  [t_win_nondim, R_exp_win_nondim, t_end_nondim] = Find_t_and_R(R_nondim, t_nondim);
% tspan_nondim = t_end_nondim;
% figure
% plot(t_win_nondim, R_exp_win_nondim)

[~, index_start] = max(R_nondim);

tspan_nondim = t_nondim(end);

% Fit from the maximum radius to the first collapse minimum.
% The old code used index_start:28, which only kept a few points near Rmax
% for this dataset and did not include the collapse/rebound behavior.
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
 Gp = 1.3e4;     % Ground-state shear modulus (Pa) %%%%% TODO
 mup = 0.12;     % Viscosity (Pa-s) %%%%% TODO

 var2fit0 = [Gp,mup];

%U0p_lb = 40;     U0p_ub = 120;
G_lb = 1e3;      G_ub = 1e5;
mu_lb = 0.01;  mu_ub = 1;
lb_primary = [G_lb, mu_lb];
ub_primary = [G_ub, mu_ub];
x0_primary = log10(var2fit0);
lb_primary_log = log10(lb_primary);
ub_primary_log = log10(ub_primary);

% =========================================================================
% DEBUG LOG SETUP
% =========================================================================
log_stamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
fid_eval = fopen(sprintf('ps_debug_eval_%s.csv', log_stamp), 'w');
fprintf(fid_eval, 'call,G,mu,log10G,log10mu,LSQErr,elapsed_s\n');

fid_iter = fopen(sprintf('ps_debug_iter_%s.csv', log_stamp), 'w');
fprintf(fid_iter, 'iter,G_best,mu_best,log10G_best,log10mu_best,LSQErr_best,meshsize,funccount,method\n');

% Reset the persistent call counter inside the wrapper
ps_log_eval([], [], []);

% Wrap objective to record every single function evaluation
orig_obj = @(x) fun_LSQErr_primary_Bubble2(10.^x, ...
    t_exp, tspan_nondim, t_nondim, ...
    Req_m, R_nondim, Rmax_um, ...
    t_win_s, R_exp_win_m);
logged_obj = @(v) ps_log_eval(v, orig_obj, fid_eval);

% OutputFcn records one row per patternsearch iteration
global PS_LOG_FID_ITER;
PS_LOG_FID_ITER = fid_iter;
% =========================================================================

% Patternsearch options used for this debug run.
options = optimoptions('patternsearch',...
        'Display','iter', ...
        'Algorithm','classic', ...          % clearer "Successful/Unsuccessful Poll" logging
        'UseCompletePoll',true, ...
        'PollMethod','GSSPositiveBasis2N', ...
        'ScaleMesh', false, ...
        'InitialMeshSize',0.3, ...
        'MeshExpansionFactor',2.0, ...
        'MeshContractionFactor',0.5, ...
        'FunctionTolerance', 1e-6, ...
        'MeshTolerance',1e-4, ...
        'StepTolerance',1e-4, ...
        'Cache','off','CacheTol',1e-12, ...
        'UseParallel', false, ...        % disabled for logging (file IO incompatible with workers)
        'OutputFcn', @ps_log_iter);    % <-- only addition to options


[var2fit_best_primary, fval, exitflag, output] = patternsearch( ...
    logged_obj, ...                    % <-- wrapped objective (same math)
    x0_primary, [], [], [], [], lb_primary_log, ub_primary_log, [], options);

var2fit_best_primary = 10.^var2fit_best_primary;
var2fit_best_primary_log = log10(var2fit_best_primary);

% =========================================================================
% SAVE DEBUG LOG
% =========================================================================
fclose(fid_eval);
fclose(fid_iter);

eval_log = readtable(sprintf('ps_debug_eval_%s.csv', log_stamp));
iter_log = readtable(sprintf('ps_debug_iter_%s.csv', log_stamp));

% Capture key search settings for side-by-side comparison with Python
search_settings = struct( ...
    'initial_guess_G',          var2fit0(1), ...
    'initial_guess_mu',         var2fit0(2), ...
    'lb_G',                     lb_primary(1), ...
    'ub_G',                     ub_primary(1), ...
    'lb_mu',                    lb_primary(2), ...
    'ub_mu',                    ub_primary(2), ...
    'initial_guess_log10G',     x0_primary(1), ...
    'initial_guess_log10mu',    x0_primary(2), ...
    'lb_log10G',                lb_primary_log(1), ...
    'ub_log10G',                ub_primary_log(1), ...
    'lb_log10mu',               lb_primary_log(2), ...
    'ub_log10mu',               ub_primary_log(2), ...
    'optimizer_space',          'log10', ...
    'Algorithm',                'classic', ...
    'UseCompletePoll',          true, ...
    'PollMethod',               'GSSPositiveBasis2N', ...
    'ScaleMesh',                false, ...
    'InitialMeshSize',          0.3, ...
    'MeshExpansionFactor',      2.0, ...
    'MeshContractionFactor',    0.5, ...
    'FunctionTolerance',        1e-6, ...
    'MeshTolerance',            1e-4, ...
    'StepTolerance',            1e-4, ...
    'Cache',                    'off', ...
    'CacheTol',                 1e-12, ...
    'UseParallel',              false, ...
    'index_start',              index_start, ...
    'index_end',                index_end, ...
    'fit_window_points',        numel(t_win_s), ...
    'fit_window_t_start_s',     t_win_s(1), ...
    'fit_window_t_end_s',       t_win_s(end), ...
    'fit_window_R_start_m',     R_exp_win_m(1), ...
    'fit_window_R_end_m',       R_exp_win_m(end), ...
    'Req_m',                    Req_m, ...
    'Rmax_um',                  Rmax_um, ...
    'Pinf',                     Pinf, ...
    'rho',                      rho, ...
    'c_long',                   c_long, ...
    'gamma',                    gamma, ...
    'px2um',                    px2um, ...
    'fps',                      fps);

save(sprintf('ps_debug_%s.mat', log_stamp), ...
    'eval_log', 'iter_log', 'search_settings', ...
    'var2fit_best_primary', 'var2fit_best_primary_log', ...
    'fval', 'exitflag', 'output');

fprintf('\n=== Debug log saved: ps_debug_%s.mat ===\n', log_stamp);
fprintf('  Total function evaluations : %d\n', height(eval_log));
fprintf('  Total iterations           : %d\n', height(iter_log));
fprintf('  Best G  = %.4e Pa\n', var2fit_best_primary(1));
fprintf('  Best mu = %.4e Pa*s\n', var2fit_best_primary(2));
fprintf('  Best LSQErr = %.4f\n', fval);
% =========================================================================


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
plot(t_shift_nondim, R_nondim, 'LineWidth',2)
legend('Sim', 'Exp', 'Fontsize', 20)

xlabel('time (nondim)','FontSize',20)
ylabel('R/R_{eq}','FontSize',20)

ax = gca;
ax.FontSize = 20;


%%
% var = [6e3 0.3];
% 
% tspan_nondim_2 = tspan2fit_exp/tc;
% [t_sim, t_sim_nondim, R_sim_nondim] = ...
% fun_Bubbles_fitting_results(var, ...
%     t_nondim, tspan_nondim_2, ...
%     Req_m, Rmax_um);
% 
% [Rsim_max, Index_Rsim_max] = max(R_sim_nondim);
% time_sim_shift = t_sim - t_sim(Index_Rsim_max);
% 
% figure
% plot(time_sim_shift, R_sim_nondim,'LineWidth',2)
% hold on
% plot(t_exp_shift, R_nondim, 'LineWidth',2)
% legend('Sim', 'Exp', 'Fontsize', 20)
% 
% xlabel('time (s)','FontSize',20)
% ylabel('R/R_{eq}','FontSize',20)
% 
% ax = gca;
% ax.FontSize = 20;

%xlim([-0.3,3.5]*10^-11)

%%
% var_best_fit = [1e5 0.1 40];
% tspan2fit_exp =  tspan2fit_exp;
% [t_sim,R_sim,U_sim,P_sim, t_sim_nondim,R_sim_nondim] = ...
% fun_IMR_damage(var_best_fit, t_nondim, R_nondim, Rmax_um, Req_exp, tspan2fit_exp);
% figure
% plot(t_sim, R_sim)
%


% =========================================================================
% LOCAL FUNCTIONS  (must be at end of script file, MATLAB R2016b+)
% =========================================================================

function err = ps_log_eval(v, obj_fn, fid)
% Called for every objective function evaluation.
% ps_log_eval([], [], []) resets the internal call counter.
% v is in log10 optimizer space; write both SI values and log10 values.
    persistent n;
    if isempty(n) || isempty(v)
        n = 0;
        err = 0;
        return
    end
    n = n + 1;
    t0 = tic;
    err = obj_fn(v);
    elapsed = toc(t0);
    v_si = 10.^v;
    fprintf(fid, '%d,%.10e,%.10e,%.6f,%.6f,%.10e,%.4f\n', ...
        n, v_si(1), v_si(2), v(1), v(2), err, elapsed);
end


function [stop, options, changed] = ps_log_iter(optimValues, options, flag)
% patternsearch OutputFcn: called once per iteration.
% Records best point, mesh size, and poll result.
% optimValues.x is in log10 optimizer space; write both SI and log10 values.
    global PS_LOG_FID_ITER;
    stop    = false;
    changed = false;
    if strcmp(flag, 'iter') && ~isempty(PS_LOG_FID_ITER)
        x = optimValues.x;
        x_si = 10.^x;
        fprintf(PS_LOG_FID_ITER, '%d,%.10e,%.10e,%.6f,%.6f,%.10e,%.4e,%d,"%s"\n', ...
            optimValues.iteration, ...
            x_si(1), x_si(2), x(1), x(2), ...
            optimValues.fval, ...
            optimValues.meshsize, ...
            optimValues.funccount, ...
            optimValues.method);
    end
end
