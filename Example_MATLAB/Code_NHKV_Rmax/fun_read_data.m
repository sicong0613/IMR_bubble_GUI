%% function for data extraction and non-dimensionalization

function [t_exp_shift, t_shift_nondim, t_exp, t_nondim, tc,  ...
    R_nondim, Rmax_um, Req_m, R_m, fileName] ...
    = fun_read_data(px2um, fps, Pinf, rho, path_bubble_Rt)

FolderPath_bubble_Rt = path_bubble_Rt;
% uigetdir('*.*', 'Select the folder containing data file (small bubbles)');

if FolderPath_bubble_Rt == 0
    disp('Invalid folder path.');
    return;
end

% choose a .mat file
[FileName, FilePath] = uigetfile(fullfile(FolderPath_bubble_Rt, '*.mat'), ...
    'Select ONE bubble data file');
[~, name, ~] = fileparts(FileName); 
fileName = name;


if isequal(FileName,0)
    disp('User cancelled file selection.');
    return;
end

fullFileName = fullfile(FilePath, FileName);
fprintf('Loading: %s\n', fullFileName);

Rdata = load(fullFileName);

%% Non-dimensionalization scale

Uc   = sqrt(Pinf/rho);

%%

% Raw radius signal (pixels)
%R0 = Rdata.CircleR(:);      % force column vector
R0 = Rdata.Radius(:);      % force column vector

% OPTIONAL: if you want to force using only the first 256 frames, uncomment:
% R0 = R0(1:min(256, numel(R0)));

% Build time axis based on the actual data length (IMPORTANT FIX)
N = numel(R0);

idx0 = 3;
% Guard: idx0 must be within bounds
if idx0 > N
    warning('idx0=%d exceeds data length N=%d in %s. Skipping.', idx0, N, inputName);
end

% Optional: force the first 3 points to zero (your current practice)
if N >= 3
    R0(1:3) = 0;
end

% Define start index for analysis segment:
idx_start = 4;

if idx_start > N
end

% Working segment (raw, NOT smoothed)
R_raw = R0(idx_start:end);
t_raw = (0:N-1-(idx_start-1)).' / fps;

% Ensure R_raw and t_raw have exactly the same length (extra safety)
L = min(numel(R_raw), numel(t_raw));
R_raw = R_raw(1:L);
t_raw = t_raw(1:L);

% Remove NaNs / Infs (now sizes match for sure)
valid = isfinite(R_raw) & isfinite(t_raw);
R_raw = R_raw(valid);
t_raw = t_raw(valid);

if numel(R_raw) < 6
end





%% -------------------- Use polynomial fitting near the experimental maximum --------------------
% Find raw experimental maximum first
[~, idx_pk_raw] = max(R_raw);

% Number of fitting points around the peak
% Example: use 5 points total: idx_pk_raw-3 : idx_pk_raw+1
i1 = max(1, idx_pk_raw-3);
i2 = min(numel(R_raw), idx_pk_raw+1);

% If near boundary and fewer than 5 points are available,
% try to expand the window to still use ~5 points if possible
targetNum = 5;
while (i2 - i1 + 1) < targetNum
    if i1 > 1
        i1 = i1 - 1;
    elseif i2 < numel(R_raw)
        i2 = i2 + 1;
    else
        break
    end
end

t_fit = t_raw(i1:i2);
R_fit = R_raw(i1:i2);

% Quadratic fit: R = p(1)*t^2 + p(2)*t + p(3)
p = polyfit(t_fit, R_fit, 2);

% Vertex of parabola gives sub-frame tRmax
tRmax = -p(2) / (2*p(1));

% Evaluate fitted maximum radius
Rmax_px = polyval(p, tRmax);

% Fallback: if fit is not physically reasonable, use raw maximum
if p(1) >= 0 || tRmax < min(t_fit) || tRmax > max(t_fit)
    warning('Quadratic fit around peak is not reliable. Falling back to raw maximum.');
    tRmax = t_raw(idx_pk_raw);
    Rmax_px = R_raw(idx_pk_raw);
end

%% -------------------- Insert (tRmax, Rmax) into the RAW R-t curve --------------------
tol = 1e-15;

if any(abs(t_raw - tRmax) < tol)

    t_aug = t_raw;
    R_aug = R_raw;
else
    j = find(t_raw < tRmax, 1, 'last');

    if isempty(j)

        t_aug = [tRmax; t_raw];
        R_aug = [Rmax_px; R_raw];

    elseif j == numel(t_raw)

        t_aug = [t_raw; tRmax];
        R_aug = [R_raw; Rmax_px];

    else

        t_aug = [t_raw(1:j); tRmax; t_raw(j+1:end)];
        R_aug = [R_raw(1:j); Rmax_px; R_raw(j+1:end)];

    end
end

%% -------------------- Shift time so that t=0 at Rmax and discard t<0 --------------------
t_exp = t_aug;
t_exp_shift = t_aug - tRmax;

%% -------------------- Non-dimensionalize --------------------


R_m = R_aug * px2um * 1e-6;

Req_px = mean(R_aug(end-29:end));
R_nondim = R_aug / Rmax_px;

Req_m = Req_px*px2um*1e-6;

Rmax_um = Rmax_px * px2um;
Rmax_m = Rmax_um * 1e-6;


tc = Rmax_m / Uc;

t_nondim = t_exp / tc;
t_shift_nondim = t_exp_shift / tc;



end

% use polynomial fitting (five points) to fit the max point of r1

