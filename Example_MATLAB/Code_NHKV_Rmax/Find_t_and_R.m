function [t_win_nondim, R_exp_win_nondim, t_end_nondim] = Find_t_and_R(R_nondim_exp, t_nondim_exp)
% ---------- Align experimental time so that R≈1 occurs at t=0 ----------
% 1) find t1: the experimental time where R_nondim is closest to 1
% [~, idx1] = min(abs(R_nondim_exp - 1));
% t1 = t_nondim_exp(idx1);

t1 = 0; %%%%%%%%YZ

% 2) shift time axis left by t1
t_exp_shift_nondim = t_nondim_exp - t1;

% 3) keep only t >= 0 (new axis)
mask = (t_exp_shift_nondim >= 0);
t_exp_use_nondim = t_exp_shift_nondim(mask);
R_exp_use_nondim = R_nondim_exp(mask);

% remove NaNs / zeros if needed

t_exp_use_nondim = t_exp_use_nondim(:);
R_exp_use_nondim = R_exp_use_nondim(:);

n = min(length(t_exp_use_nondim), length(R_exp_use_nondim));

t_exp_use_nondim = t_exp_use_nondim(1:n);
R_exp_use_nondim = R_exp_use_nondim(1:n);

valid = isfinite(t_exp_use_nondim) & isfinite(R_exp_use_nondim);

t_exp_use_nondim = t_exp_use_nondim(valid);
R_exp_use_nondim = R_exp_use_nondim(valid);

%% ---------- Define window: equilibrium -> expansion -> first collapse minimum ----------
% find first local minimum (collapse) after a short initial window
minSearchStart = 3;

[~, idxMinRel] = min(R_exp_use_nondim(minSearchStart:end));
idxMin = idxMinRel + (minSearchStart - 1);

t_end_nondim = t_exp_use_nondim(idxMin);

% 3) crop the window [0, t_end]
maskWin = (t_exp_use_nondim >= 0) & (t_exp_use_nondim <= t_end_nondim);
t_win_nondim = t_exp_use_nondim(maskWin);
R_exp_win_nondim = R_exp_use_nondim(maskWin);



end