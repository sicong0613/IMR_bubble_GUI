%% Setup and load data
clear; clc; close all;


% box_path = "C:\Users\jy24967\Box\research_projects\Weekly_individual_report\";
box_path = "C:\Users\49531\Box";

um2px = 3.2; % 3.2 um per pixel

fps = 1e6; % frames per second

Pinf = 101325; % unit: Pa
rho = 1000;
cav_depth = 542e-6; %  m
tspan2fit_exp = 100e-6; % 38.5e-6; % unit: s


% read data
father_path = "Code_Rmax_as_beginning";
path_file = fullfile(father_path, "05mW_01min_01_physical");
option_read_data = [path_file, um2px, fps, Pinf, rho, cav_depth];
[t_exp, R_exp, t_exp_nondim, R_exp_nondim, gamma, Rmax_exp, R_eq, tc] = fun_read_data(option_read_data);

% initialize the variables to fit
mat_name1 = "Fitting_Result_Guess\struct_DNA_origami_guess_0530_164234.mat";
mat_name2 = "Fitting_Result_Guess\struct_DNA_0530_161428_fit_best.mat";
% leave the input empty to use the default initial guess, go to the function fitting_initialization to see the default initial guess
% input an existing mat file to use the initial guess in the mat file
struct_vars = fun_fitting_initialization(mat_name1);

time_now = datestr(now, 'mm_dd_HH_MM');
mat_name = "Fitting_Result_Guess\struct_DNA_0530_164234_fit_best.mat";

%% optimization

% all_optimization(struct_vars, t_exp, R_exp, t_exp_nondim, R_exp_nondim, R_eq, Rmax_exp, tc, mat_name);

%% test
% [t_sim, R_sim, U_sim, P_sim, t_sim_nondim, R_sim_nondim] = fun_IMR_damage_PoyntingThomson_Kelvin_KM(var2fit0, R1eq_exp);
% plot(t_sim_nondim, R_sim_nondim, '--');
% hold on; plot(t_exp_nondim, R1_exp_nondim, 's');

%% result plot
% does not record the xi_mesh and other messy parameters
mat_name_result = 'simulation_result\DNA_0530_160946';
save_option = struct('save_name', mat_name_result, 'diag_record', false,  'is_save', false);


result_plot(mat_name, tspan2fit_exp, R_eq, t_exp_nondim, R_exp_nondim, t_exp, R_exp, 1, save_option);


function [] = all_optimization(struct_vars, t_exp, R_exp, t_nondim_exp, R_nondim_exp, Req_exp, Rmax_exp, tc, mat_name)
    % The difficulty of this part is that the time and radius scale of simulation is usually different from the experimental data
    % But the normalized simulated result may be a good fit to the experimental data
    % This indicates that the tspan_exp we input is not exactly what the simulation should take
    % which is why we multiply the tspan_exp by a certain number to ensure the simulation can reach the peak

    %% First, find the first valley point of R(1)_nondim_exp
    R_ind = find(t_exp > 1.6e-5, 1, 'first');
    R_nondim_toFit_exp = [R_nondim_exp(1: R_ind)];
    t_nondim_toFit_exp = [t_nondim_exp(1: R_ind)];

    tspan2fit_exp = (t_exp(R_ind) - t_exp(1)) * 1.1;          % tspan for simulation should be large enough to simulate the peak

    [var2fit0,lb,ub, map] = make_search_space(struct_vars, 'groups', {'expansion1'});

    options = optimoptions('patternsearch',...
        'Display','iter', ...
        'Algorithm','classic', ...          % clearer ¡°Successful/Unsuccessful Poll¡± logging
        'UseCompletePoll',true, ...
        'PollMethod','GSSPositiveBasis2N', ...
        'ScaleMesh',false, ...              % you already log-transform
        'InitialMeshSize',0.1, ...
        'MeshContractionFactor',0.4, ...
        'MeshTolerance',1e-5, ...
        'StepTolerance',1e-10, ...
        'Cache','on','CacheTol',1e-12, ...
        'UseParallel', true);


    %% debug
    % err = LSQErr_all(var2fit0, map, tspan2fit_exp, t_nondim_toFit_exp, R_nondim_toFit_exp, Req_exp, tc);
    % disp(err);

    %% optimize
    [var_best_fit,fval,exitflag,output] = patternsearch( @(var2fit) ...
        LSQErr_all(var2fit, map, tspan2fit_exp, t_nondim_toFit_exp, R_nondim_toFit_exp, Req_exp, tc), ...
        var2fit0, [],[],[],[],lb,ub, [ ], options);

    struct_best_fit = vec_to_struct(var_best_fit, map);

    save(mat_name, 'struct_best_fit');
end

function [LSQErr] = LSQErr_all(var2fit, map, tspan2fit_exp, t_nondim_toFit_exp, R_nondim_toFit_exp, R_eq, tc)
    % This function call fun_IMR_damage_PorntingThomson_Kelvin.m to simulate the expansion process
    % and then only take the expansion part to compute the Least Squares Error
    % var_other: other variables that are not to be fitted

    var2fit = assemble_parameters(var2fit, map);

    % show_bar = false;

    % diag_record = false;

    % Input: var2fit, R1eq_exp, tspan2fit_exp
    try
        [t_sim, R_sim, U_sim, P_sim, t_sim_nondim, R_sim_nondim] = fun_IMR_NHKV(var2fit, t_nondim_toFit_exp, R_nondim_toFit_exp, R_eq, tspan2fit_exp);
    catch
        % If simulation fails, return a large error value
        LSQErr = 1e10;
        return;
    end

    % Check if simulation returned valid results
    if isempty(t_sim) || isempty(R_sim) || length(t_sim) < 2 || length(R_sim) < 2
        LSQErr = 1e10;
        return;
    end

    % restore the normalized experimental data to the original scale
    R_toFit_exp = R_nondim_toFit_exp * R_eq;
    t_toFit_exp = t_nondim_toFit_exp * tc;

    LSQErr = fun_LSQErr_interpolation(t_sim, R_sim, t_toFit_exp, R_toFit_exp);
end


function [] = result_plot(mat_name, t_span, R_eq, t_nondim_exp, R_nondim_exp, t_exp, R_exp, fig_num, save_option)
    load(mat_name);
    vars = struct_to_var(struct_best_fit);

    % show_bar = true;

    % diag_record = save_option.diag_record;

    [t_sim, R_sim, U_sim, P_sim, t_sim_nondim, R_sim_nondim] = fun_IMR_NHKV(vars, t_nondim_exp, R_nondim_exp, R_eq, t_span);

    figure(fig_num);
    plot(t_exp, R_exp, 's', 'color', 'k', 'markerfacecolor', 'k');
    hold on

    plot(t_sim, R_sim, '--', 'LineWidth', 1.4, 'color', 'r');
    hold on

    legend('Exp', 'fit');
    grid on;
    title('RT curve');
    xlabel('t (s)');
    ylabel('R (m)');
    set(gca, 'FontSize', 14);

    xlim([-3, 8.5]*1e-5);
    % xlim([-0.5, 2.5]*1e-5);
    % ylim([0, 2.25]*1e-4);


    % figure
    % plot(t_nondim_exp, R_nondim_exp, 's', 'color', 'k', 'markerfacecolor', 'k');
    % hold on
    % plot(t_sim_nondim, R_sim_nondim, '--', 'LineWidth', 1.4, 'color', 'r');
    % hold off
    % 
    % legend('Exp', 'fit');
    % xlim([-5, 5])
    % grid on;
    % title('RT curve (normalized)');
    % xlabel('t*');
    % ylabel('R*');
    % set(gca, 'FontSize', 14);

    % xlim([-0.5, 3]);
    % ylim([0, 2]);
    % 
    % figure(fig_num + 1);
    % plot(diag.t_nondim, diag.WA_wall, 'lineWidth', 2);
    % hold on;
    % plot(diag.t_nondim, diag.WA_max, 'lineWidth', 2);
    % hold off;
    % legend('WA_wall', 'WA_max');
    % grid on;
    % title('WA');
    % xlabel('t*');
    % ylabel('WA');
    % set(gca, 'FontSize', 14);

    Time_1 = 2.8e-5;
    Time_3 = 6e-5;

    t_ind_1 = find(t_exp > Time_1, 1, 'first');
    t_ind_3 = find(t_exp > Time_3, 1, 'first');

    t_exp_1 = t_exp(1: t_ind_1);
    R1_exp_1 = R_exp(1: t_ind_1);
    t_exp_3 = t_exp(1: t_ind_3);
    R1_exp_3 = R_exp(1: t_ind_3);
    LSQErr_1 = fun_LSQErr_interpolation(t_sim, R_sim, t_exp_1, R1_exp_1);
    % LSQErr_3 = fun_LSQErr_interpolation(t_sim, R_sim, t_exp_3, R1_exp_3);

    if save_option.is_save == true
        save(save_option.save_name, 't_sim', 'R_sim', 'U_sim', 'P_sim', 't_sim_nondim', 'R_sim_nondim', 't_exp', 't_nondim_exp', 'R_exp', 'R_nondim_exp', 'struct_best_fit');
    end

    var_best_fit = struct_to_var(struct_best_fit);
    disp([newline, 'U0: ', num2str(var_best_fit(1))]);
    disp(['G: ', num2str(var_best_fit(2))]);
    disp(['mu: ', num2str(var_best_fit(3))]);
    disp(['Least Squares Error as of the first cycle is: ', num2str(LSQErr_1)]);
    % disp(['Least Squares Error as of the third cycle is: ', num2str(LSQErr_3)]);

end

function [var2fit0,lb,ub, map] = make_search_space(struct_vars, varargin)
    % varargin: e.g., 'names', {'k1','k2','E1'}  or  'groups', {'kinetics','elastic'}
    sel = false(1,numel(struct_vars));
    
    if any(strcmp(varargin, 'names'))
        names = varargin{find(strcmp(varargin,'names'))+1};
        struct_names = string({struct_vars.name});
        for i = 1:numel(names), sel = sel | strcmp(struct_names, string(names{i})); end
    elseif any(strcmp(varargin, 'groups'))
        groups = varargin{find(strcmp(varargin,'groups'))+1};
        struct_groups = string({struct_vars.group});
        for i = 1:numel(groups), sel = sel | strcmp(struct_groups, string(groups{i})); end
    else
        error('Specify ''names'', {...} or ''groups'', {...}');
    end
    
    idx = find(sel);
    
    % Build decision vector in mixed (log/lin) search space
    var2fit0 = zeros(numel(idx),1); lb = var2fit0; ub = var2fit0;
    for j = 1:numel(idx)
        r = struct_vars(idx(j));
        if r.scale == "log"
            var2fit0(j) = log10(r.value); lb(j) = log10(r.lb); ub(j) = log10(r.ub);
        else
            var2fit0(j) = r.value;      lb(j) = r.lb;      ub(j) = r.ub;
        end
    end
    
    % map helper carries indices & scales to reconstruct full theta
    map.idx = idx;
    map.scale = string({struct_vars(idx).scale});
    map.names = string({struct_vars(idx).name});
    map.template = struct_vars; % keep original for frozen params
end

function vars = assemble_parameters(var2fit0, map)
    % Start from template default values (frozen params untouched)
    struct_vars = map.template;
    
    % Overwrite only the active ones
    for j = 1:numel(map.idx)
        k = map.idx(j);
        if map.scale(j) == "log"
            struct_vars(k).value = 10 ^ (var2fit0(j));
        else
            struct_vars(k).value = var2fit0(j);
        end
    end
    
    % Convert registry array -> plain matrix assigned by the order of [U0, GA1, GA2, alpha1, alpha2, GB1, GB2, beta1, beta2, mu, damage_n, damage_m, damage_k]
    vars = zeros(1, numel(struct_vars));
    for k = 1:numel(struct_vars)
        vars(k) = struct_vars(k).value;
    end
end

function struct_best_fit = vec_to_struct(var_best_fit, map)
    % Build a full parameter struct with fields by name
    % using map.template as the base (frozen params preserved)
        struct_vars = map.template;            % full registry
        for j = 1:numel(map.idx)
            k = map.idx(j);            % position in full registry
            if map.scale(j) == "log"
                struct_vars(k).value = 10.^var_best_fit(j);
            else
                struct_vars(k).value = var_best_fit(j);
            end
        end
        % registry array -> struct with fields
        struct_best_fit = struct_vars;
end

function vars = struct_to_var(struct_best_fit)
% (Optional) Extract active decision vector from a struct theta
% respecting log10 vs linear scales defined in map
    vars = zeros(numel(struct_best_fit),1);
    for j = 1:numel(struct_best_fit)
        vars(j) = struct_best_fit(j).value;
    end
end
