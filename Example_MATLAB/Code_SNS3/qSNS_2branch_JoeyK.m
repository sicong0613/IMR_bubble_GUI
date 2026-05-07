%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IMR-EIC (TWO-branch) : single-pass fitting of G, G1, mu1, G2, mu2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clear; clc;
%% 0) User settings
dataFile = '20241202_001_high_R_data';  % -> must contain CircleR
um2px    = 17.0357751277683;                % [um/px]%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
FPS      = 100e3;                           % [Hz]
T_inf    = 80 + 273.15;                     % [K] ambient
% Physical constants
gamma    = 0.0725;   % [N/m]
C_long   = 1119.867; % [m/s] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%TODO  [1110.605, 1130.79, 1119.867, 1226.517]
rho      = 1060;     % [kg/m^3]
P_inf    = 101325;   % [Pa]
fprintf('## 1. 데이터 로딩 및 전처리... ##\n');
load(dataFile);                        % -> loads CircleR
t_exp_s = (1:numel(CircleR)) / FPS;    % [s]
R_exp_m = CircleR * um2px * 1e-6;      % [m]
t_full  = (min(t_exp_s):0.1/FPS:max(t_exp_s))';
R_full  = pchip(t_exp_s, R_exp_m, t_full);
fprintf('   데이터 로딩 및 보간 완료.\n');
%% 2) Pick Req, t* offset and span; nondimensionalize
fprintf('\n## 2. 데이터 정규화 및 피팅 범위 설정... ##\n');
% -- Pick Req on a flat tail point --
figure('Name','Select Equilibrium Radius (Req)','NumberTitle','off');
plot(t_full*1e3, R_full*1e6, 'b.-'); grid on;
xlabel('Time (ms)'); ylabel('Radius (\mum)');
title('Click one stable point to set R_{eq}');
fprintf('>> 평형 반경(Req) 지점 1회 클릭:\n');
[~, y_click_um] = ginput(1);
Req = y_click_um * 1e-6;  % [m]
close(gcf);
fprintf('   Req = %.2f um\n', Req*1e6);
% -- nondimensionalize with req --
Uc = sqrt(P_inf/rho);
tc = Req/Uc;
t_star_raw = t_full / tc;
R_star     = R_full / Req;
% -- Pick start (t*=0) and end (=> tspan*) --
figure('Name','Select Time Zero and Fit Span','NumberTitle','off');
plot(t_star_raw, R_star, 'b.-'); grid on;
xlabel('Original t/t_c'); ylabel('R/R_{eq}');
title({'Select 2 points:','(1) Start of expansion, (2) End of fit window'});
fprintf('\n>> 시작점과 종료점을 순서대로 클릭:\n');
[x_points,~] = ginput(2);
close(gcf);
t_star_0   = x_points(1);
t_end_raw  = x_points(2);
tspan_star = t_end_raw - t_star_0;
if tspan_star <= 0
    error('End must be after start. Click start first, then end.');
end
fprintf('   t_star_0 = %.4f,  tspan_star = %.4f\n', t_star_0, tspan_star);
% -- shift and export for record --
t_star = t_star_raw - t_star_0;
t      = t_star * tc;
R      = R_star * Req;
save('data_for_fitting_nd.mat','t','R','t_star','R_star','Req','T_inf','t_star_0','tspan_star');
fprintf('   최종 데이터 준비 완료. (data_for_fitting_nd.mat)\n');
%% 3) Single-pass optimization: fit [G, G1, mu1, G2, mu2]
fprintf('\n## 3. 2-Branch 최적화 (G, G1, mu1, G2, mu2)... ##\n');
fit_mask = (t_star >= 0 & t_star < tspan_star);
idx_fit  = find(fit_mask);
if numel(idx_fit) < 10
    error('피팅 구간 데이터가 너무 적습니다. tspan_star를 더 길게 선택하세요.');
end
t_fit = t(idx_fit);
R_fit = R(idx_fit);
R0    = R(idx_fit(1));  % first point at t*>=0
% Initial guesses and bounds (log10 space) for 5 variables
% 섹션 3의 초기값/경계 (log10)% [G,G1,mu1,G2,mu2]
x0 = [ log10(2e6), log10(5e6), log10(3e-3), log10(6e5), log10(5e-2) ];
lb = [ log10(1e3), log10(1e5), log10(1e-4), log10(1e5), log10(1e-4) ];
ub = [ log10(5e7), log10(5e9), log10(3e-1), log10(5e7), log10(3e+1) ];

% ---- (섹션 3) x0/lb/ub 교체 ----
% x0 = [ log10(1e6), log10(1e6), log10(1e-2), log10(1e6), log10(1e-1) ]; %
% lb = [ log10(1e5), log10(1e4), log10(1e-3), log10(1e4), log10(1e-3)];
% ub = [ log10(5e7), log10(5e7), log10(3e1),  log10(5e7), log10(3e1)];

obj = @(v) rmse_two_branch(v, t_fit, R_fit, R0, Req, T_inf, gamma, rho, C_long);
opts = optimoptions(@patternsearch, ...
    'InitialMeshSize', 1, ...
    'UseCompletePoll', true, ...
    'Display', 'iter', ...
    'TolX', 1e-4, ...
    'UseParallel', true);
global TERM_HISTORY;
TERM_HISTORY = [];
[var_best, fval] = patternsearch(obj, x0, [],[],[],[], lb, ub, [], opts);
best_params.G   = 10.^var_best(1);
best_params.G1  = 10.^var_best(2);
best_params.mu1 = 10.^var_best(3);
best_params.G2  = 10.^var_best(4);
best_params.mu2 = 10.^var_best(5);
fprintf('\n✅ 최적화 완료!\n');
fprintf('=====================================\n');
fprintf('  G   = %.3e Pa\n', best_params.G);
fprintf('  G1  = %.3e Pa, mu1 = %.4f Pa-s\n', best_params.G1, best_params.mu1);
fprintf('  G2  = %.3e Pa, mu2 = %.4f Pa-s\n', best_params.G2, best_params.mu2);
fprintf('  Final RMSE = %.4e\n', fval);
fprintf('=====================================\n\n');

%% 4) Final simulation & plots (full length)
% --- [MODIFIED SECTION START] ---
fprintf('## 4. 최종 결과 시뮬레이션 및 시각화... ##\n');
clRGB = [240, 176, 61] / 255; % Orange color for experimental data

% === NEW: output folder for .fig files ===
outDir = fullfile(pwd, ['IMR_plots_' datestr(now,'yyyymmdd_HHMMSS')]);
if ~exist(outDir,'dir'), mkdir(outDir); end

% Run the final simulation
full_tspan_star = max(t_star);
full_tspan = full_tspan_star * tc;
TERM_HISTORY = []; % Reset for final simulation run
[t_sim, R_sim] = simulate_IMR_EIC_twobranch(best_params, R0, Req, T_inf, gamma, rho, C_long, full_tspan);
t_sim_star = t_sim / tc;
R_sim_star = R_sim / Req;

% --- Plot 1: Dimensional Overlay ---
fh1 = figure('Name', 'Final Overlay (Dimensional)', 'Color', 'w');
plot(t * 1e3, R * 1e6, 'o', 'Color', clRGB, 'MarkerFaceColor', clRGB, 'MarkerSize', 4, 'DisplayName', 'Experiment');
hold on;
plot(t_sim * 1e3, R_sim * 1e6, 'k-', 'LineWidth', 2, 'DisplayName', 'SnLS fit');
xlabel('Time $t$ (ms)', 'Interpreter', 'latex');
ylabel('$R$ ($\mu$m)', 'Interpreter', 'latex');
legend('Location', 'best');
set(gca, 'FontSize', 24, 'LineWidth', 1);
grid on;
savefig(fh1, fullfile(outDir, 'plot1_dimensional.fig'));

% --- Plot 2: Normalized Overlay (Full Timespan) ---
fh2 = figure('Name', 'Final Overlay (Normalized - Full)', 'Color', 'w');
plot(t_star, R_star, 'o', 'Color', clRGB, 'MarkerFaceColor', clRGB, 'MarkerSize', 4, 'DisplayName', 'Experiment');
hold on;
plot(t_sim_star, R_sim_star, 'k-', 'LineWidth', 2, 'DisplayName', 'SnLS fit');
xlabel('Normalized time $t^*$', 'Interpreter', 'latex');
ylabel('Normalized $R^*$', 'Interpreter', 'latex');
legend('Location', 'best');
set(gca, 'FontSize', 24, 'LineWidth', 1);
grid on;
xlim([-full_tspan_star/2, full_tspan_star]); % Adjust x-axis to show pre-zero time
savefig(fh2, fullfile(outDir, 'plot2_normalized_full.fig'));

% --- Plot 3: Normalized Overlay (Zoomed In) ---
fh3 = figure('Name', 'Final Overlay (Normalized - Zoomed)', 'Color', 'w');
plot(t_star, R_star, 'o', 'Color', clRGB, 'MarkerFaceColor', clRGB, 'MarkerSize', 4, 'DisplayName', 'Experiment');
hold on;
plot(t_sim_star, R_sim_star, 'k-', 'LineWidth', 2, 'DisplayName', 'SnLS fit');
xlabel('Normalized time $t^*$', 'Interpreter', 'latex');
ylabel('Normalized $R^*$', 'Interpreter', 'latex');
legend('Location', 'best');
set(gca, 'FontSize', 24, 'LineWidth', 1);
grid on;
xlim([0, full_tspan_star]); % Set x-axis to start from 0
ylim([min(R_star(t_star>0))*0.9, max(R_star)*1.1]); % Auto-adjust y-axis
savefig(fh3, fullfile(outDir, 'plot3_normalized_zoom.fig'));
% --- [MODIFIED SECTION END] ---

%% 5) Keller-Miksis 방정식 구성 요소 시각화
fprintf('## 5. Keller-Miksis 방정식 구성 요소 시각화... ##\n');
if isempty(TERM_HISTORY) || size(TERM_HISTORY, 1) < 2
    fprintf('!! WARNING: TERM_HISTORY is empty. Final simulation likely failed.\n');
else
    fh4 = figure('Name', 'Keller-Miksis Equation Terms (2-Branch)', 'Color', 'w');
    time_history = TERM_HISTORY(:,1);
    pressure_term = TERM_HISTORY(:,2);
    surfacetension_term = TERM_HISTORY(:,3);
    elastic_term = TERM_HISTORY(:,4);
    viscoelastic1_term = TERM_HISTORY(:,5);
    viscoelastic2_term = TERM_HISTORY(:,6);
    plot(time_history, pressure_term, 'DisplayName', 'Pressure (P^*-1)'); hold on;
    plot(time_history, surfacetension_term, 'DisplayName', 'Surface Tension');
    plot(time_history, elastic_term, 'DisplayName', 'Elastic (S_{eq})');
    plot(time_history, viscoelastic1_term, 'DisplayName', 'Viscoelastic (S_{neq1})');
    plot(time_history, viscoelastic2_term, 'DisplayName', 'Viscoelastic (S_{neq2})');
    grid on;
    legend('Location','best');
    titleString = '$R^*\ddot{R}^* (1-\frac{\dot{R}^*}{C^*}) = (1+\frac{\dot{R}^*}{C^*})(P^*... -1) + \frac{R^*}{C^*}\frac{d(P^*...)}{dt^*} - \frac{3}{2}{\dot{R}^*}^2(1-\frac{\dot{R}^*}{3C^*})$';
    title(titleString, 'Interpreter', 'latex', 'FontSize', 10);
    xlabel('Normalized time t*');
    ylabel('Nondimensional Pressure/Stress');
    savefig(fh4, fullfile(outDir, 'plot4_KM_terms.fig'));
end
clear global TERM_HISTORY;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Local functions (two-branch IMR-EIC)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% --- FIXED: Renamed function to rmse_two_branch and updated it to handle 5 parameters ---
function val = rmse_two_branch(v, t_fit, R_fit, R0, Req, T_inf, gamma, rho, C_long)
global TERM_HISTORY;
TERM_HISTORY = [];
try
    % Unpack 5 parameters
    pars.G   = 10.^v(1);
    pars.G1  = 10.^v(2);
    pars.mu1 = 10.^v(3);
    pars.G2  = 10.^v(4);
    pars.mu2 = 10.^v(5);
    
    tspan_fit = max(t_fit);
    
    % Call the two-branch simulator
    [t_num, R_num] = simulate_IMR_EIC_twobranch(pars, R0, Req, T_inf, gamma, rho, C_long, tspan_fit);
    
    if isempty(t_num) || any(~isfinite(R_num)) || numel(t_num) < 10
        val = 1e12; return;
    end
    R_model = interp1(t_num, R_num, t_fit, 'linear', 'extrap');
    err = R_model - R_fit;
    val = sqrt(mean(err.^2));
    if ~isfinite(val) || ~isreal(val)
        val = 1e12;
    end
catch
    val = 1e12;
end
end

function [t, R] = simulate_IMR_EIC_twobranch(pars, R0, Req, T_inf, gamma, rho, c_long, tspan)
global TERM_HISTORY;
D0=24.2e-6; kappa=1.4; Ru=8.3144598; Rv=Ru/(18.01528e-3); Ra=Ru/(28.966e-3);
A=5.28e-5; B=1.17e-2; P_ref=1.17e11; T_ref=5200; P_inf=101325;
NT=100; MT=100; IMRsolver_RelTolX=1e-5;
Uc=sqrt(P_inf/rho); Rc=Req; tc=Rc/Uc; C_star=c_long/Uc;
We=P_inf*Rc/(2*gamma); Ca=P_inf/pars.G;
Ca1=P_inf/pars.G1; Ca2=P_inf/pars.G2;
Re1=P_inf*Rc/(pars.mu1*Uc); Re2=P_inf*Rc/(pars.mu2*Uc);
De1=pars.mu1*Uc/(pars.G1*Rc); De2=pars.mu2*Uc/(pars.G2*Rc);
Pv=P_ref*exp(-T_ref/T_inf); K_inf=A*T_inf+B;
fom=D0/(Uc*Rc); chi=T_inf*K_inf/(P_inf*Rc*Uc);
A_star=A*T_inf/K_inf; B_star=B/K_inf; Pv_star=Pv/P_inf;
tspan_star=tspan/tc; R0_star=R0/Rc; U0_star=0; alpha=0;
P0=Pv+(P_inf+2*gamma/R0-Pv)*((Req/R0)^3); P0_star=P0/P_inf;
Seq0=(3*alpha-1)*(5-4*(Req/R0)-(Req/R0)^4)/(2*Ca)+2*alpha*(27/40+1/8*(Req/R0)^8+1/5*(Req/R0)^5+1*(Req/R0)^2-2/(Req/R0))/(Ca);
Theta0=zeros(1,NT); k0=((1+(Rv/Ra)*(P0_star/Pv_star-1))^(-1))*ones(1,NT);
s_neq_rr1_0=zeros(1,MT); s_neq_rr2_0=zeros(1,MT);
X0=[R0_star U0_star P0_star Seq0 Theta0 k0 s_neq_rr1_0 s_neq_rr2_0]';
params=[NT C_star We Ca Ca1 Ca2 alpha Re1 Re2 De1 De2 Rv Ra kappa fom chi A_star B_star Pv_star 1 0 0 0 0 MT];
opts=odeset('RelTol',IMRsolver_RelTolX);
[t_star_sol,X_sol]=ode15s(@(t,X) bubble_twobranch(t,X,params),[0 tspan_star],X0,opts);
% [t_star_sol,X_sol]=ode23tb(@(t,X) bubble_twobranch(t,X,params),[0 tspan_star],X0,opts);
t=t_star_sol*tc;
R=X_sol(:,1)*Rc;
end

function dxdt = bubble_twobranch(t,x,params)
global TERM_HISTORY;
NT=params(1); C_star=params(2); We=params(3); Ca=params(4); Ca1=params(5);
Ca2=params(6); alpha=params(7); Re1=params(8); Re2=params(9); De1=params(10);
De2=params(11); Rv=params(12); Ra=params(13); kappa=params(14); fom=params(15);
chi=params(16); A_star=params(17); B_star=params(18); Pv_star=params(19);
Req=params(20); PA_star=params(21); omega_star=params(22); delta_star=params(23);
n=params(24); MT=params(25);
R=x(1); U=x(2); P=x(3); Seq=x(4);
Theta=x(5:(NT+4)); k=x((NT+5):(2*NT+4));
s_neq_rr1=x((2*NT+5):(2*NT+4+MT));
s_neq_rr2=x((2*NT+5+MT):(2*NT+4+2*MT));
temp_array=linspace(0,2,MT); r0_star_list=10.^(temp_array(:));
deltaY=1/(NT-1); yk=((0:NT-1)*deltaY)';
k(end)=(1+(Rv/Ra)*(P/Pv_star-1))^(-1);
T=(A_star-1+sqrt(1+2.*A_star.*Theta))./A_star; K_star=A_star.*T+B_star; Rmix=k.*Rv+(1-k).*Ra;
DTheta=[0;(Theta(3:end)-Theta(1:end-2))/(2*deltaY);(3*Theta(end)-4*Theta(end-1)+Theta(end-2))/(2*deltaY)];
DDTheta=[6*(Theta(2)-Theta(1))/deltaY^2;(diff(diff(Theta)/deltaY)/deltaY+(2./yk(2:end-1)).*DTheta(2:end-1));((2*Theta(end)-5*Theta(end-1)+4*Theta(end-2)-Theta(end-3))/deltaY^2+(2/yk(end))*DTheta(end))];
Dk=[0;(k(3:end)-k(1:end-2))/(2*deltaY);(3*k(end)-4*k(end-1)+k(end-2))/(2*deltaY)];
DDk=[6*(k(2)-k(1))/deltaY^2;(diff(diff(k)/deltaY)/deltaY+(2./yk(2:end-1)).*Dk(2:end-1));((2*k(end)-5*k(end-1)+4*k(end-2)-k(end-3))/deltaY^2+(2/yk(end))*Dk(end))];
pdot=3/R*(-kappa*P*U+(kappa-1)*chi*DTheta(end)/R+kappa*P*fom*Rv*Dk(end)/(R*Rmix(end)*(1-k(end))));
Umix=((kappa-1).*chi./R.*DTheta-R.*yk.*pdot./3)./(kappa.*P)+fom./R.*(Rv-Ra)./Rmix.*Dk;
Theta_prime=(pdot+(DDTheta).*chi./R.^2).*(K_star.*T./P.*(kappa-1)./kappa)-DTheta.*(Umix-yk.*U)./R+fom./(R.^2).*(Rv-Ra)./Rmix.*Dk.*DTheta;
Theta_prime(end)=0;
k_prime=fom./R.^2.*(DDk+Dk.*(-((Rv-Ra)./Rmix).*Dk-DTheta./sqrt(1+2.*A_star.*Theta)./T))-(Umix-U.*yk)./R.*Dk;
k_prime(end)=0;
lambda_w=R/Req;
Seqdot=2*U/R*(3*alpha-1)*(1/lambda_w+1/lambda_w^4)/Ca-2*alpha*U/R*(1/lambda_w^8+1/lambda_w^5+2/lambda_w^2+2*lambda_w)/(Ca);
lambda_out=(1+(1./r0_star_list).^3.*(lambda_w^3-1)).^(1/3);
lambdadot_out=lambda_w^2*U./r0_star_list.^3./lambda_out.^2;
term_r=lambda_out.*r0_star_list;
S_neq1=trapz(term_r,3./term_r.*s_neq_rr1);
s_neq_rr1_dot=(-4/Re1*lambdadot_out./lambda_out-s_neq_rr1)/De1;
Sdot1_term1=3*U*R^2./(term_r.^4).*s_neq_rr1;
Sdot1_term2=3./term_r.*s_neq_rr1_dot;
S_neq1_dot=-trapz(term_r,Sdot1_term1)+trapz(term_r,Sdot1_term2)-3/R*s_neq_rr1(1)*U;
S_neq2=trapz(term_r,3./term_r.*s_neq_rr2);
s_neq_rr2_dot=(-4/Re2*lambdadot_out./lambda_out-s_neq_rr2)/De2;
Sdot2_term1=3*U*R^2./(term_r.^4).*s_neq_rr2;
Sdot2_term2=3./term_r.*s_neq_rr2_dot;
S_neq2_dot=-trapz(term_r,Sdot2_term1)+trapz(term_r,Sdot2_term2)-3/R*s_neq_rr2(1)*U;
Pext=0; Pextdot=0;
rdot=U;
udot=((1+U/C_star)*(P-1/(We*R)+Seq+S_neq1+S_neq2-1-Pext)+R/C_star*(pdot+U/(We*R^2)+Seqdot+S_neq1_dot+S_neq2_dot-Pextdot)-(3/2)*(1-U/(3*C_star))*U^2)/((1-U/C_star)*R); %KM

 % udot = ( (P - 1/(We*R) + Seq + S_neq1 + S_neq2 - 1 - Pext) - (3/2)*U^2 ) / ( R ); %RP
if isempty(TERM_HISTORY) || t > TERM_HISTORY(end,1)
    term_PressureDrive = P-1;
    term_SurfaceTension = -1/(We*R);
    term_Elastic = Seq;
    new_row = [t, term_PressureDrive, term_SurfaceTension, term_Elastic, S_neq1, S_neq2];
    TERM_HISTORY = [TERM_HISTORY; new_row];
end
dxdt=[rdot;udot;pdot;Seqdot;Theta_prime;k_prime;s_neq_rr1_dot;s_neq_rr2_dot];
end
