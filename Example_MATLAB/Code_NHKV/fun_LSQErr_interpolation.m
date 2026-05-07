
function [LSQErr] = fun_LSQErr_interpolation(t_sim, R_sim, t_toFit_exp, R1_toFit_exp)
    % find the first peak point of R_sim_nondim, but a little bit later, since the peak in experimental data is a little bit later than 0
    R_sim_max_ind = find(t_sim > t_toFit_exp(end), 1, 'first');
    R_sim_expansion = R_sim(1: R_sim_max_ind);
    t_sim_expansion = t_sim(1: R_sim_max_ind);

    % Check if expansion arrays are valid for interpolation
    if isempty(t_sim_expansion) || isempty(R_sim_expansion) || length(t_sim_expansion) < 2
        LSQErr = 1e10;
        return;
    end

    % check point 1
    % figure,
    % plot(t_toFit_exp, R1_toFit_exp, 's');
    % hold on; plot(t_sim_expansion, R_sim_expansion, '--');

    % interpolate the simulated result to the experimental data
    try
        R_sim_interp = interp1(t_sim_expansion, R_sim_expansion, t_toFit_exp, 'makima', 100);
    catch
        % If interpolation fails, return a large error value
        LSQErr = 1e10;
        return;
    end

    % debug
    R1_toFit_exp = R1_toFit_exp(4: end);
    R_sim_interp = R_sim_interp(4: end);
    t_toFit_exp = t_toFit_exp(4: end);

    % check point 2
    % figure,
    % plot(t_toFit_exp, R1_toFit_exp, 's');
    % hold on; plot(t_toFit_exp, R_sim_interp, 'o');

    % compute the Least Squares Error
    % In case of error, change the dimension to um here
    R1_toFit_exp = R1_toFit_exp * 1e6;
    R_sim_interp = R_sim_interp * 1e6;
    LSQErr = sum((R1_toFit_exp - R_sim_interp).^2);
end