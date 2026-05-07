function [t_sim,R_sim,U_sim,P_sim, t_sim_nondim, R_sim_nondim, diag] = fun_IMR_damage_PoyntingThomson_Kelvin_KM(var2fit, Req_exp, tspan, show_bar, diag_record, xi_constant)

    %%%%% Far-field thermodynamic conditions %%%%%
    P_inf = 101325; % Atmospheric pressure (Pa)
    T_inf = 298.15; % Far field temperature (K)
    % Req_exp = 100e-6;

    R0 = Req_exp; % R0: initial radius (m)

    %%%%% Type of cavitation driving force %%%%%
    %     LIC = Laser induced cavitation
    %     UIC = Ultrasound induced cavitation
    cav_type = 'LIC'; %%%%% TODO


    %%%%% Variables to fit %%%%%
    % Whether parameters should be fitted or not is decided by the function that calls this function
    U0       = var2fit(1);       % initial speed, to fit (m/s)
    GA1      = var2fit(2);       % unit: Pa
    GA2      = var2fit(3);       % unit: Pa
    alpha1   = var2fit(4);  
    alpha2   = var2fit(5);
    GB1      = var2fit(6);       % unit: Pa
    GB2      = var2fit(7);       % unit: Pa
    beta1    = var2fit(8);
    beta2    = var2fit(9);
    mu       = var2fit(10);      % unit: Pa-s

    damage_index = var2fit(11);

    matpara.GA1 = GA1; matpara.GA2 = GA2; matpara.alpha1 = alpha1; matpara.alpha2 = alpha2;
    matpara.GB1 = GB1; matpara.GB2 = GB2; matpara.beta1 = beta1; matpara.beta2 = beta2;
    matpara.mu = mu;
    matpara.damage_index = damage_index;

    %%%%% Driving conditions %%%%%
    if strcmp(cav_type,'LIC') == 1

        % Rc = R0; % Maximum (initial) bubble radius for LIC (m) %%%%% TODO
        % U0 = 100; % initial speed, to fit
        PA = 0; omega = 0; delta = 0; n = 0; % These values won't be used in LIC

    elseif strcmp(cav_type,'UIC') == 1

        PA = -24e6; % Amplitude of the ultrasound pulse (Pa) %%%%% TODO
        omega = 2*pi*(1e6); % Frequency of the ultrasound pulse (1/s) %%%%% TODO
        delta = pi/omega; % Time shift for the ultrasound pulse (s) %%%%% TODO
        n = 3.7; % Exponent that shapes the ultrasound pulse %%%%% TODO
        % Rmax = R0; % Assume that initial bubble radius is R0 (stress free)

    else
        disp('Incorrect cavitation type');
        return;

    end

    %%%%% Total simulation time %%%%%
    % tspan = 200e-6; % Total time span (s)

    %%%%% Material parameters for the surrounding material %%%%%
    % G = 2.77e3;     % Ground-state shear modulus (Pa) %%%%% TODO
    % alpha = 0.48;   % Strain-stiffening parameter (1) (alpha=0: neo-Hookean) %%%%% TODO
    % mu = 0.186;     % Viscosity (Pa-s) %%%%% TODO
    % c_long = 1430;  % Longitudinal wave speed (m/s)
    % rho = 1060;     % Density (kg/m^3)
    % gamma = 5.6e-2; % Surface tension (N/m)
    alpha = 0; % 0.48;   % Strain-stiffening parameter (1) (alpha=0: neo-Hookean) %%%%% TODO

    c_long =  1485;  % Longitudinal wave speed (m/s)
    rho = 998;     % Density (kg/m^3)
    gamma = 5.6e-2; % Surface tension (N/m)





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
    MT = 200; % Mesh points outside the bubble, resolution should be >xxx
    IMRsolver_RelTolX = 1e-7; % Relative tolerance for the ode solver




    %%
    % Intermediate calculated variables
    %
    % General characteristic scales
    if strcmp(cav_type,'LIC') == 1
        Rc = Req_exp; % Characteristic length scale (m)
        Uc = sqrt(P_inf/rho); % Characteristic velocity (m/s)
        tc = Req_exp/Uc; % Characteristic time scale (s)
    elseif strcmp(cav_type,'UIC') == 1
        Rc = R0; % Characteristic length scale (m)
        Uc = sqrt(P_inf/rho); % Characteristic velocity (m/s)
        tc = R0/Uc; % Characteristic time scale (s)
    end

    % tspan = 250e-6; % tspan2fit_exp;

    % Parameters for the bubble contents
    Pv = P_ref*exp(-T_ref./T_inf); % Vapor pressure evaluated at the far field temperature (Pa)
    K_inf = A*T_inf+B; % Thermal conductivity evaluated at the far field temperature (W/m-K)

    %%
    % Non-dimensional variables
    %
    C_star = c_long/Uc; % Dimensionless wave speed
    We = P_inf*Rc/(2*gamma); % Weber number
    CaA1 = P_inf/GA1; % Cauchy number
    CaA2 = P_inf/GA2; % Cauchy number
    CaB1 = P_inf/GB1; % Cauchy number
    CaB2 = P_inf/GB2; % Cauchy number
    Re = P_inf*Rc/(mu*Uc); % Reynolds number
    fom = D0/(Uc*Rc); % Mass Fourier number
    chi = T_inf*K_inf/(P_inf*Rc*Uc); % Lockhart–Martinelli number
    A_star = A*T_inf/K_inf; % Dimensionless A parameter
    B_star = B/K_inf; % Dimensionless B parameter (Note that A_star+B_star=1.)
    Pv_star = Pv/P_inf; % Dimensionless vapor saturation pressure at the far field temperature
    %
    tspan_star = tspan/tc; % Dimensionless time span
    %
    %%%%% Non-dimensional variable only used for LIC %%%%%
    Req_nondim = 1; % Dimensionless equilibrium bubble radius

    %%%%% Non-dimensional variables only used for UIC %%%%%
    PA_star = PA/P_inf; % Dimensionless amplitude of the ultrasound pulse
    omega_star = omega*tc; % Dimensionless frequency of the ultrasound pulse
    delta_star = delta/tc; % Dimensionless time shift for the ultrasound pulse

    global H;
    H.achieveRmaxOrNot = 0;
    H.achievedLambdaA = zeros(MT,1);
    H.WA_m = 0; % store maximum loaded strain energy
    H.S = 0; % stress integral (elastic part) S(t)
    H.t = 0; % nondimensional time points
    H.Sdot_current = 0;
    H.xi_constant = xi_constant; % Sicong 09.22: whether xi should be set to 0

    if diag_record == true
        H.diag_record = true;

        % Sicong 10.01: added some new variables to track the history of WA, xi, etc.
        H.WA_wall   = [];           % WA at bubble wall (index end) vs time
        H.WA_max    = [];     % max(WA_0) over all MT grid points vs time
        H.WA_max_index = 0;   % index of max(WA_0) over all MT grid points vs time
        H.xi_mesh_history = [];    % xi mesh history
        H.t_nondim = []; % nondimensional time points where WA was recorded

        % recording all the times points is expensive, so only record the time points every 0.01 seconds
        H.record_time = 0; % time point last recorded
        H.this_record = false; % whether to record the current time point
        H.record_interval = 0.01; % record interval (s)
    else
        H.diag_record = false;
        H.this_record = false; % whether to record the current time point
    end

    % Place the necessary quantities in a parameters vector
    params = [NT MT C_star We CaA1 CaA2 CaB1 CaB2 alpha1 alpha2 beta1 beta2 Re Rv Ra kappa fom chi ...
        A_star B_star Pv_star Req_nondim PA_star omega_star delta_star n ...
        damage_index ];


    %%
    % Initial conditions
    %
    R0_star = 1; % Dimensionless initial bubble radius
    U0_star = U0/Uc; % Dimensionless initial bubble wall velocity
    Theta0 = zeros(1,NT); % Initial dimensionless temperature field
    %
    if strcmp(cav_type,'LIC') == 1
        P0 = Pv + (P_inf + 2*gamma/R0 - Pv)*((R0/Req_exp)^3); % Initial bubble pressure for LIC
        P0_star = P0/P_inf; % Dimensionless initial bubble pressure for LIC
        % S0 = ...
        %   (3*alpha-1)*(5 - 4*Req_nondim - Req_nondim^4)/(2*Ca) + ...
        %    2*alpha*(27/40 + 1/8*Req_nondim^8 + 1/5*Req_nondim^5 + 1*Req_nondim^2 - 2/Req_nondim)/(Ca); % Initial dimensionless elastic stress integral for LIC
        k0 = ((1+(Rv/Ra)*(P0_star/Pv_star-1))^(-1))*ones(1,NT); % Initial vapor mass fraction for LIC
        lambda_nv0 = 1.00001*ones(1,MT); % Initial lambda_nv (non-equilibrium branch)
    elseif strcmp(cav_type,'UIC') == 1
        P0 = P_inf + 2*gamma/R0; % Initial bubble pressure for UIC
        P0_star = P0/P_inf; % Dimensionless initial bubble pressure for UIC
        Se0 = 0; % Initial dimensionless elastic stress integral for UIC
        k0 = ((1+(Rv/Ra)*(P0_star/Pv_star-1))^(-1))*ones(1,NT); % Initial vapor mass fraction for UIC
    end

    % Place the initial conditions in the state vector
    X0 = [R0_star U0_star P0_star Theta0 k0 lambda_nv0];
    X0 = reshape(X0,length(X0),1);

    %% Solve the system of ODEs
    %
    if strcmp(cav_type,'LIC') == 1
        IMRsolver_InitialStep = [];
    elseif strcmp(cav_type,'UIC') == 1
        % Make sure that the initial time step is sufficiently small for UIC
        IMRsolver_InitialStep = delta_star/20;
    end
    if show_bar == true
        maxWallTimeSec = [];  % input 120 to stop after 2 minutes if still running, leave [] to not stop
        options = odeset('RelTol',IMRsolver_RelTolX,'InitialStep',IMRsolver_InitialStep, ...
                'OutputFcn', @(t,y,flag) fun_odeprogressbar(t,y,flag,[0 tspan_star],maxWallTimeSec), 'Stats','on');
    else
        options = odeset('RelTol',IMRsolver_RelTolX,'InitialStep',IMRsolver_InitialStep);
    end
    % tic;
    [t_nondim,X_nondim] = ode23tb(@(t,X) bubble(t,X,cav_type,params),[0 tspan_star], X0, options);
    % toc

    % Extract the solution
    R_nondim = X_nondim(:,1); % Bubble wall radius history
    U_nondim = X_nondim(:,2); % Bubble wall velocity history
    P_nondim = X_nondim(:,3); % Internal bubble pressure history
    Theta_nondim = X_nondim(:,4:(NT+3)); % Variable relating to internal temp (theta)
    k_nondim = X_nondim(:,(NT+4):(2*NT+3)); % Vapor concentration in the bubble
    lambda_nv = X_nondim(:,(2*NT+4):(2*NT+3+MT));

    T_nondim = (A_star - 1 + sqrt(1+2.*A_star.*Theta_nondim))./A_star; % Dimensionless temp in bubble


    %% Result

    %%%%%% Back to physical world %%%%%%%
    t_sim = t_nondim*tc; % unit: s
    R_sim = R_nondim*Rc; % unit: m
    U_sim = U_nondim*(Rc/tc); % unit: m/s
    P_sim = P_nondim*P_inf; % unit: Pa

    % shift the time to the peak point of R_sim
    [Rmax_sim, Rmax_sim_ind] = max(R_sim); 
    t_sim_shifted = t_sim - t_sim(Rmax_sim_ind);
    t_sim = t_sim_shifted;

    R_sim_nondim = R_sim / Rc;
    t_sim_nondim = t_sim * Uc / Rc;

    diag = struct();
    % ---- Build diagnostic struct ----
    diag.achievedLambdaA = H.achievedLambdaA;      % The lambda_A value at damage_index
    diag.achievedLambdaB = H.achievedLambdaB;      % The lambda_B value at damage_index
    if diag_record == true
        diag.t_nondim        = H.t_nondim;             % raw nondim time stamps where WA was recorded
        diag.WA_wall         = H.WA_wall;              % WA at bubble wall
        diag.WA_m_snapshot   = H.WA_m;                 % the WA_m vector captured at R_max (element-wise)
        diag.WA_max_index    = H.WA_max_index;         % index of max(WA_0) over all MT grid points vs time
        diag.xi_mesh_history = H.xi_mesh_history;      % xi mesh history
        diag.lambdaA_history = H.lambdaA_history;      % The lambda_A value at damage_index
        diag.lambdaB_history = H.lambdaB_history;      % The lambda_B value at damage_index
    else
        diag.t_nondim = [];
        diag.WA_wall = [];
        diag.WA_max = [];
        diag.WA_m_snapshot = [];
        diag.WA_max_index = [];
        diag.xi_mesh_history = [];
        diag.lambdaA_history = [];
        diag.lambdaB_history = [];
    end
end



%%
%*************************************************************************
% Function that the ODE Solver calls to march governing equations in time
% Unlike in vanilla-IMR this is NOT a nested function
function dxdt = bubble(t,x,cav_type,params)

    % Extract quantities from the parameters vector
    NT = params(1); % Mesh points inside the bubble
    MT = params(2); % Mesh points outside the bubble
    C_star = params(3); % Dimensionless wave speed
    We = params(4); % Weber number
    CaA1 = params(5); % Cauchy number
    CaA2 = params(6); % Cauchy number
    CaB1 = params(7); % Cauchy number
    CaB2 = params(8); % Cauchy number
    alpha1 = params(9); %
    alpha2 = params(10);
    beta1 = params(11);
    beta2 = params(12);
    Re = params(13); % Reynolds number
    Rv = params(14); % Gas constant for vapor (J/kg-K)
    Ra = params(15); % Gas constant for air (J/kg-K)
    kappa = params(16); % Specific heats ratio
    fom = params(17); % Mass Fourier number
    chi = params(18); % Lockhart–Martinelli number
    A_star = params(19); % Dimensionless A parameter
    B_star = params(20); % Dimensionless B parameter (Note that A_star+B_star=1.)
    Pv_star = params(21); % Dimensionless vapor saturation pressure at the far field temperature
    Req = params(22); % Dimensionless equilibrium bubble radius (LIC only)
    PA_star = params(23); % Dimensionless amplitude of the ultrasound pulse (UIC only)
    omega_star = params(24); % Dimensionless frequency of the ultrasound pulse (UIC only)
    delta_star = params(25); % Dimensionless time shift for the ultrasound pulse (UIC only)
    n = params(26); % Exponent that shapes the ultrasound pulse (UIC only)

    damage_index = params(27);

    % Se_f = 1/Ca/2*(5-4/lambda_Y-1/lambda_Y^4);

    % Extract quantities from the state vector
    R = x(1); % Bubble wall radius
    U = x(2); % Bubble wall velocity
    P = x(3); % Internal bubble pressure
    Theta = x(4:(NT+3)); % Variable relating to internal temp (theta)
    k = x((NT+4):(2*NT+3)); % Vapor mass fraction (k)
    lambda_nv = x((2*NT+4):(2*NT+3+MT)); % Non-equilibrium branch: lambdaA


    %******************************************
    % Set up grid outside the bubble
    temp_array = linspace(0,3,MT);
    r0_star_list = 10.^(temp_array(:)); % Dimensionless grid points outside the bubble
    %******************************************

    %******************************************
    % Set up grid inside the bubble
    deltaY = 1/(NT-1); % Dimensionless grid spacing inside the bubble
    ii = 1:1:NT;
    yk = ((ii-1)*deltaY)'; % Dimensionless grid points inside the bubble
    %******************************************

    %******************************************
    % Apply the Dirichlet BC for the vapor mass fraction at the bubble wall
    k(end) = (1+(Rv/Ra)*(P/Pv_star-1))^(-1);
    %******************************************

    %******************************************
    % Calculate mixture fields inside the bubble
    T = (A_star - 1 + sqrt(1+2.*A_star.*Theta))./A_star; % Dimensionless temperature T/T_inf
    if ~isreal(T)
        disp('error: T is not a real number');
        T = (A_star - 1 + sqrt(1+2.*A_star.*Theta))./A_star; % Dimensionless temperature T/T_inf
    end
    K_star = A_star.*T+B_star; % Dimensionless mixture thermal conductivity field
    Rmix = k.*Rv + (1-k).*Ra; % Mixture gas constant field (J/kg-K)
    %******************************************

    %******************************************
    % Calculate spatial derivatives of the temp and vapor conc fields
    DTheta = [0; % Neumann BC at origin
        (Theta(3:end)-Theta(1:end-2))/(2*deltaY); % Central difference approximation for interior points
        (3*Theta(end)-4*Theta(end-1)+Theta(end-2))/(2*deltaY)]; % Backward difference approximation at the bubble wall
    DDTheta = [6*(Theta(2)-Theta(1))/deltaY^2; % Laplacian in spherical coords at the origin obtained using L'Hopital's rule
        (diff(diff(Theta)/deltaY)/deltaY + (2./yk(2:end-1)).*DTheta(2:end-1)); % Central difference approximation for Laplacian in spherical coords
        ((2*Theta(end)-5*Theta(end-1)+4*Theta(end-2)-Theta(end-3))/deltaY^2+(2/yk(end))*DTheta(end))]; % Laplacian at the bubble wall does not affect the solution
    Dk = [0; % Neumann BC at origin
        (k(3:end)-k(1:end-2))/(2*deltaY); % Central difference approximation for interior points
        (3*k(end)-4*k(end-1)+k(end-2))/(2*deltaY)]; % Backward difference approximation at the bubble wall
    DDk = [6*(k(2)-k(1))/deltaY^2; % Laplacian in spherical coords at the origin obtained using L'Hopital's rule
        (diff(diff(k)/deltaY)/deltaY + (2./yk(2:end-1)).*Dk(2:end-1)); % Central difference approximation for Laplacian in spherical coords
        ((2*k(end)-5*k(end-1)+4*k(end-2)-k(end-3))/deltaY^2+(2/yk(end))*Dk(end))]; % Laplacian at the bubble wall does not affect the solution
    %******************************************

    %******************************************
    % Internal bubble pressure evolution equation
    pdot = 3/R*(-kappa*P*U + (kappa-1)*chi*DTheta(end)/R ...
        + kappa*P*fom*Rv*Dk(end)/(R*Rmix(end)*(1-k(end))));
    %******************************************

    %******************************************
    % Dimensionless mixture velocity field inside the bubble
    Umix = ((kappa-1).*chi./R.*DTheta-R.*yk.*pdot./3)./(kappa.*P) + fom./R.*(Rv-Ra)./Rmix.*Dk;
    %******************************************

    %******************************************
    % Evolution equation for the temperature (theta) of the mixture inside the bubble
    Theta_prime = (pdot + (DDTheta).*chi./R.^2).*(K_star.*T./P.*(kappa-1)./kappa) ...
        - DTheta.*(Umix-yk.*U)./R ...
        + fom./(R.^2).*(Rv-Ra)./Rmix.*Dk.*DTheta;
    Theta_prime(end) = 0; % Dirichlet BC at the bubble wall
    %******************************************

    %******************************************
    % Evolution equation for the vapor concentration inside the bubble
    k_prime = fom./R.^2.*(DDk + Dk.*(-((Rv - Ra)./Rmix).*Dk - DTheta./sqrt(1+2.*A_star.*Theta)./T)) ...
        - (Umix-U.*yk)./R.*Dk;
    k_prime(end) = 0; % Dirichlet BC at the bubble wall
    %******************************************

    %******************************************
    % Elastic stress in the material
    %     (viscous contribution is accounted for in the Keller-Miksis equation)
    if strcmp(cav_type,'LIC') == 1

        lambda_w = R/Req;  % circumferential stretch ratio at the bubble wall

        r0_star_list3 = r0_star_list.^3;

        lambda_r0 = ( 1 + (1./r0_star_list3).*(lambda_w^3-1) ).^(1/3);  % lambda ratio is position varying

        term_r = lambda_r0.*r0_star_list;   % current radial distance r(r0,t)

        % lambdadot_out = ( 1./ (1 + r0_star_list3/lambda_w^3 - 1/lambda_w^3)).^(2/3)*U/Req./r0_star_list ;

        lambda_ne = lambda_r0 ./ lambda_nv;        % lambda elastic portion of branch Bn
        lambda_ne_power_beta1 =  lambda_ne.^beta1; % store it to save computational cost
        lambda_ne_power_beta2 =  lambda_ne.^beta2; % store it to save computational cost

        s_tt_nv = 1/CaB1/3*( lambda_ne_power_beta1 - lambda_ne_power_beta1.^(-2) ) + ...
            1/CaB2/3*( lambda_ne_power_beta2 - lambda_ne_power_beta2.^(-2) );

        lambda_nv_dot = s_tt_nv /2*Re .* lambda_nv; % lambda viscous portion of branch Bn


        % -------------------
        lambdaA_power1 = lambda_r0.^alpha1;
        if CaA2<1e10,  lambdaA_power2 = lambda_r0.^alpha2; end
        % If GA2 is very large, CaA2 can be ignored to save computational time

        if CaA2<1e10
            WA_0 = 1/CaA1/alpha1* ( (lambdaA_power1).^(-2) + 2*(lambdaA_power1) - 3 ) + ...
                1/CaA2/alpha2* ( (lambdaA_power2).^(-2) + 2*(lambdaA_power2) - 3 );
        else % If GA2 is very large, CaA2 can be ignored to save computational time
            WA_0 = 1/CaA1/alpha1* ( (lambdaA_power1).^(-2) + 2*(lambdaA_power1) - 3 );
        end

        % determine whether the parameters are stored based on t
        global H
        if H.diag_record == true
            if H.record_time == 0 && length(H.t) == 1
                H.this_record = true;  % initial time point
                H.record_time = t;
            elseif abs(H.record_time - t) < H.record_interval
                H.this_record = false;
            else
                H.this_record = true;
                H.record_time = t;
            end
        end


        if H.achieveRmaxOrNot==0 % bubble has not achieved Rmax yet (no damage yet)

            xi = ones(MT,1); % no damage

            if H.this_record == true
                H.WA_max_index = H.WA_max_index + 1;
                H.xi_mesh_history = [H.xi_mesh_history; xi'];
            end

            if U < abs(eps) % criterion to know bubble reaches Rmax

                H.R_max = R;    % store nondimensional R_max
                H.achieveRmaxOrNot = 1;
                H.achievedLambdaA = lambda_r0;
                H.achievedLambdaB = lambda_ne;
                H.WA_m = WA_0; % store maximum loaded strain energy

            end

        else
            % -------------------------------
            % Binary xi by (lambda>1)
            % -------------------------------

            % Nodal current radius and stretch (already computed earlier this step)
            lam     = lambda_r0;         % nodal circumferential stretch

            % Condition (1): tensile
            tensile = (lam > 1);
            % tensile = true;

            % Binary xi: if the material is subjected to tensile stress, material within damage_index is damaged; otherwise 1
            xi = ones(MT,1);
            if tensile(1) == true
                xi = [zeros(damage_index, 1); ones(MT-damage_index, 1)];
            end

            if H.xi_constant >= 0
                xi = H.xi_constant * ones(MT,1);
            elseif H.xi_constant == -0.5
                xi = [zeros(damage_index, 1); ones(MT-damage_index, 1)];
            end

            if H.this_record == true
                H.xi_mesh_history = [H.xi_mesh_history; xi'];
            end
        end

        % log nondimensional time and WA summaries
        if H.this_record == true
            H.t_nondim = [H.t_nondim; t];
            H.WA_wall = [H.WA_wall; WA_0(1)];  % WA at the bubble wall
            H.lambdaA_history = [H.lambdaA_history; lambda_r0'];
            H.lambdaB_history = [H.lambdaB_history; lambda_ne'];
        end



        if CaA2<1e10
            Sint_term1 = xi .* ( 1/CaA1 *( (lambdaA_power1).^(-2) - (lambdaA_power1) ) + ...
                1/CaA2*( lambdaA_power2.^(-2) - lambdaA_power2 ) ) ;
        else
            Sint_term1 = xi .* ( 1/CaA1 *( (lambdaA_power1).^(-2) - (lambdaA_power1) ) );
        end
    
        if CaB2<1e10
            Sint_term2 = 1/CaB1*( lambda_ne.^(-2*beta1) - lambda_ne.^(beta1) ) + ...
                1/CaB2*( lambda_ne.^(-2*beta2) - lambda_ne.^(beta2) ) ;
        else
            Sint_term2 = 1/CaB1*( lambda_ne.^(-2*beta1) - lambda_ne.^(beta1) )  ;
        end
    
    
        S = trapz( term_r, 2*(Sint_term1+Sint_term2)./term_r ); % stress integral


        % Calculate Sdot
        S_store = H.S;
        t_store = H.t;
        
        if max(t_store) < abs(eps)
            Sdot = 0;
        else

            if abs(t_store(end) - t )  < abs(eps)
                Sdot = H.Sdot_current;
            else
                Sdot = ( S_store(end) - S ) / (t_store(end) - t );
                H.Sdot_current = Sdot;
            end
        end

        % Update H.t and H.S
        H.t = [H.t; t];
        H.S = [H.S; S];


    elseif strcmp(cav_type,'UIC') == 1

        Sedot = 2*U/R*(3*alpha-1)*(1/R + 1/R^4)/Ca - ...
            2*alpha*U/R*(1/R^8 + 1/R^5 + 2/R^2 + 2*R)/(Ca);

    end
    %******************************************

    %******************************************
    % External pressure for UIC
    if strcmp(cav_type,'LIC') == 1
        Pext = 0; % No external pressure for LIC
        Pextdot = 0;
    elseif strcmp(cav_type,'UIC') == 1
        if (abs(t-delta_star)>(pi/omega_star))
            Pext = 0;
            Pextdot = 0;
        else
            Pext = PA_star*((1+cos(omega_star*(t-delta_star)))/2)^n;
            Pextdot = (-omega_star*n*PA_star/2)*(((1+cos(omega_star*(t-delta_star)))/2)^(n-1))*sin(omega_star*(t-delta_star));
        end
    end
    %******************************************


    %******************************************
    % Rayleigh Plesset equation
    % rdot = U;
    % udot = ( (P - 1/(We*R) + S - 1 - Pext) - (3/2)*U^2 ) / ( R );
    % ============================================
    % Keller-Miksis equations
    rdot = U;
    udot = ((1+U/C_star)*(P - 1/(We*R) + S - 1 - Pext)  ...
        + R/C_star*(pdot + U/(We*R^2) + Sdot - Pextdot) ...
        - (3/2)*(1-U/(3*C_star))*U^2)/((1-U/C_star)*R);
    %******************************************

    dxdt = [rdot; udot; pdot; Theta_prime; k_prime; lambda_nv_dot];


end