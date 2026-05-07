function [t_sim, t_sim_nondim, R_sim_nondim] = ...
fun_Bubbles_fitting_results(var_best_fit, ...
    t_exp, tspan_nondim, ...
    Req_m, Rmax_um)

% [P_primary_nondim, t_sim_nondim, R_sim_fit_nondim] = fun_primary_Bubbles_fitting_results(var2fit_best_primary, ...
%     t_exp, tspan_nondim,  time_shifted_cav_exp_nondim, ...
%     Rp_eq, Rp_exp);


%U0 = var_best_fit(1);
U0 = 0;
G = var_best_fit(1);
mu = var_best_fit(2);
% P_shockwave = var_best_fit(3);
% omega = var_best_fit(4);
%U0_small = var2fit(4:end);

%%%%% Far-field thermodynamic conditions %%%%%
P_inf = 101325; % Atmospheric pressure (Pa)
T_inf = 298.15; % Far field temperature (K)
rho = 1000;
c_long = 1486; %  1485;  % Longitudinal wave speed (m/s)
gamma = 0.0725; % Surface tension (N/m)
alpha = 0;


%cav_type = input('Please fill in the type of cavitation driving force (LIC or UIC): ', 's');
cav_type = 'LIC';

%Dist_primary_bubble = Dist_primary_nondim * Rp_eq;

%%%%% Driving conditions %%%%%
if strcmp(cav_type,'LIC') == 1

     PA = 0; omega = 0; delta = 0; n =0;

elseif strcmp(cav_type,'UIC') == 1
    PA = -24e6; % Amplitude of the ultrasound pulse (Pa) %%%%% TODO
    omega = 2*pi*(1e6); % Frequency of the ultrasound pulse (1/s) %%%%% TODO
    delta = pi/omega; % Time shift for the ultrasound pulse (s) %%%%% TODO
    n = 3.7; % Exponent that shapes the ultrasound pulse %%%%% TODO
    Rmax = R0; % Assume that initial bubble radius is R0 (stress free)

else
    disp('Incorrect cavitation type');
    return;

end

%%

%%%%%%%%% Please DO NOT modify these parameters unless you know more %%%%%%
%%%%%%%%% accurate information about the bubble contents %%%%%%%%%%%%%%%%%%
%%%%% Parameters for the bubble contents %%%%%
D0 = 24.2e-6;           % Binary diffusion coeff (m^2/s)
kappa = 1.4;            % Specific heats ratio
Ru = 8.3144598;         % Universal gas constant (J/mol-K)
Rv = Ru/(18.01528e-3);  % Gas constant for vapor (Ru/molecular weight) (J/kg-K)
Ra = Ru/(28.966e-3);    % Gas constant for air (Ru/molecular weight) (J/kg-K)
A = 5.28e-5;            % Thermal conductivity parameter (W/m-K^2)
B = 1.17e-2;            % Thermal conductivity parameter (W/m-K)
P_ref = 1.17e11;        % Reference pressure (Pa)
T_ref = 5200;           % Reference temperature (K)

%%%%%%%%% Please DO NOT modify these parameters unless you want to achieve
%%%%%%%%% higher accuracy in the ode solver %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Numerical parameters %%%%%
NT = 500; % Mesh points inside the bubble, resolution should be >=500
IMRsolver_RelTolX = 1e-7; % Relative tolerance for the ode solver


Rmax_m = Rmax_um * 1e-6;

%%
% Intermediate calculated variables
%
% General characteristic scales
if strcmp(cav_type,'LIC') == 1
    Rc = Rmax_m; % Characteristic length scale (m)
    Uc = sqrt(P_inf/rho); % Characteristic velocity (m/s)
    tc = Rmax_m/Uc; % Characteristic time scale (s)
elseif strcmp(cav_type,'UIC') == 1
    Rc = R0; % Characteristic length scale (m)
    Uc = sqrt(P_inf/rho); % Characteristic velocity (m/s)
    tc = R0/Uc; % Characteristic time scale (s)
end

%tspan = tspan2fit_exp*1e-6; %%%%

% Parameters for the bubble contents
Pv = P_ref * exp(-T_ref./T_inf); % Vapor pressure evaluated at the far field temperature (Pa)
K_inf = A*T_inf+B; % Thermal conductivity evaluated at the far field temperature (W/m-K)


%%
% Non-dimensional variables
%
C_star = c_long/Uc; % Dimensionless wave speed
We = P_inf*Rc/(2*gamma); % Weber number
Ca = P_inf/G; % Cauchy number
Re = P_inf * Rc / (mu * Uc); % Reynolds number
fom = D0/(Uc*Rc); % Mass Fourier number
chi = T_inf*K_inf/(P_inf*Rc*Uc); % Lockhart–Martinelli number
A_star = A*T_inf/K_inf; % Dimensionless A parameter
B_star = B/K_inf; % Dimensionless B parameter (Note that A_star+B_star=1.)
Pv_star = Pv/P_inf; % Dimensionless vapor saturation pressure at the far field temperature


%%%%% Non-dimensional variable only used for LIC %%%%%
%Req_nondim = R0/Rmax; % Dimensionless equilibrium bubble radius
%R_eq_nondim = R_exp_initial / Rp_eq;


%%%%% Non-dimensional variables used for ultrasound %%%%%
PA_star = PA/P_inf; % Dimensionless amplitude of the ultrasound pulse
omega_star = omega*tc; % Dimensionless frequency of the ultrasound pulse
delta_star = delta/tc; % Dimensionless time shift for the ultrasound pulse

Req_nondim = Req_m/Rmax_m;

% Place the necessary quantities in a parameters vector
params = [NT C_star We Rv Ra kappa fom chi ...
    A_star B_star Pv_star Req_nondim  ...
    PA_star omega_star delta_star n ...
    Ca Re];


%%
% Initial conditions
%

R0_star = Rmax_m / Rmax_m;
%R0_small_star = R_initial_nondim;
Up0_star = U0/Uc;

% Initial dimensionless temperature field
Thetap0 = zeros(1,NT);
%Theta_Start = reshape(Theta0.', numel(Theta0), 1);

%
if strcmp(cav_type,'LIC') == 1

    %%%%%%%%%%%%%%%%%%%% We changed here 09/23/2025 %%%%%%%%%%%%%%%%%%%%
    P0 = Pv + (P_inf + 2*gamma/Req_m - Pv)*((Req_m/Rmax_m)^3); % Initial bubble pressure for LIC
   % Pp0 = Pv + (P_inf + 2*gamma/Rp0 - Pv)*((Rp0/Req_m)^3); % Initial bubble pressure for LIC
    Pp0_star = P0/P_inf; % Dimensionless initial bubble pressure for LIC


    % Initial dimensionless elastic stress integral for LIC
    Sep0 = (3*alpha-1)*(5 - 4*Req_nondim - Req_nondim^4)/(2*Ca) + ...
            2*alpha*(27/40 + 1/8*Req_nondim^8 + 1/5*Req_nondim^5 + 1*Req_nondim^2 - 2/Req_nondim)/(Ca); % Initial dimensionless elastic stress integral for LIC

    kp0 = ((1+(Rv/Ra)*(Pp0_star/Pv_star-1))^(-1))*ones(1,NT); % Initial vapor mass fraction for LIC

    %k0 = ((1+(Rv/Ra)*(P0_star/Pv_star-1))^(-1))*ones(1,NT); % Initial vapor mass fraction for LIC

elseif strcmp(cav_type,'UIC') == 1
    P0 = P_inf + 2*gamma/R0; % Initial bubble pressure for UIC
    P0_star = P0/P_inf; % Dimensionless initial bubble pressure for UIC
    Sep0 = 0; % Initial dimensionless elastic stress integral for UIC
    kp0 = ((1+(Rv/Ra)*(P0_star/Pv_star-1))^(-1))*ones(1,NT); % Initial vapor mass fraction for UIC
end

% Place the initial conditions in the state vector
% X0 = [R0_star U0_star P0_star Se0 Theta0 k0];
% X0 = reshape(X0,length(X0),1);

X0 = [R0_star ...
    Up0_star ...
    Pp0_star ...
    Sep0 ...
    Thetap0 ...
    kp0];

X0 = reshape(X0,length(X0),1);

tspan_star = tspan_nondim;



%% Solve the system of ODEs
%
if strcmp(cav_type,'LIC') == 1
    IMRsolver_InitialStep = [];
elseif strcmp(cav_type,'UIC') == 1
    % Make sure that the initial time step is sufficiently small for UIC
    IMRsolver_InitialStep = delta_star/20;
end
options = odeset('RelTol',IMRsolver_RelTolX,'InitialStep',IMRsolver_InitialStep);
tic
[t_nondim,X_nondim] = ode23tb(@(t,X) bubble(t,X,cav_type,params),[0 tspan_star], X0, options);
toc


%%
%%%%%%% Collect the experimental results%%%%%%
t_sim_nondim = t_nondim;
R_sim_nondim = X_nondim(:, 1);
U_sim_nondim = X_nondim(:, 2);
P_sim_nondim = X_nondim(:, 3);
Se_sim_nondim = X_nondim(:, 4);



%%%%%% Back to physical world %%%%%%%
t_sim = t_sim_nondim*tc; % unit: s
R_sim = R_sim_nondim*Rc; % unit: m
U_sim = U_sim_nondim*(Rc/tc); % unit: m/s
P_sim = P_sim_nondim*P_inf; % unit: Pa


end