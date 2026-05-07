%% function for data extraction and non-dimensionalization
function [t_exp, R1, t_nondim, R1_nondim, R21, gamma, R1max, R1_eq, tc] = fun_read_data(option_read_data)

    if length(option_read_data) ~= 6
        error("option_read_data should be [path_file, um2px, fps, Pinf, rho, cav_depth]");
    end

    % option_read_data: [path_file, um2px, fps, Pinf, rho, cav_depth]
    path = option_read_data(1);
    um2px = double(option_read_data(2));
    fps = double(option_read_data(3));
    Pinf = double(option_read_data(4));
    rho = double(option_read_data(5));
    d = double(option_read_data(6));

    load(fullfile(path, "\ellipse_fitting_results.mat"));

    R1 = [ a_lengths((a_lengths > 0))]*um2px*1e-6 ; % unit: m
    R2 = [ b_lengths((b_lengths > 0))]*um2px*1e-6 ; % unit: m

    [R1max_loc, R1max] = Rmax_fit(R1); % R1max_loc's unit if frame number
    
    % find the equilibrium radius
    R1_eq = mean(R1(end-20:end));

    R1_nondim = R1 / R1_eq;
    R2_nondim = R2 / R1_eq;

    R2overR1 = R2 ./ R1;
    R21 = R2./R1;

    t_exp = [ 1 : 1 : length(R1)]/fps ; t_exp = t_exp(:); % Exp raw time points (unit:  s)
    
    Uc = sqrt(Pinf/rho);
    tc = R1_eq/Uc;

    t_R1max = (R1max_loc)/fps;
    t_nondim = (t_exp - t_R1max ) / tc; 

    gamma = d / R1max;

    % shift the time axis to the R1max point
    t_exp = t_nondim * tc;
    R1 = R1_nondim * R1_eq;
end


% use polynomial fitting (five points) to fit the max point of r1
function [Rmax_loc, Rmax] = Rmax_fit(r1)

    [~, r1_max_ind] = max(r1);
    p = polyfit(r1_max_ind-2:r1_max_ind+1, r1(r1_max_ind-2:r1_max_ind+1), 2);
    Rmax_loc = -p(2)/2/p(1);            % unit: frame #
    Rmax = (4*p(1)*p(3)-p(2)^2)/4/p(1); % unit: um

end