function dxdt = bubble(t,x,cav_type,params)

% Extract quantities from the parameters vector
NT = params(1); % Mesh points inside the bubble
C_star = params(2); % Dimensionless wave speed
We = params(3); % Weber number
Rv = params(4); % Gas constant for vapor (J/kg-K)
Ra = params(5); % Gas constant for air (J/kg-K)
kappa = params(6); % Specific heats ratio
fom = params(7); % Mass Fourier number
chi = params(8); % Lockhart–Martinelli number
A_star = params(9); % Dimensionless A parameter
B_star = params(10); % Dimensionless B parameter (Note that A_star+B_star=1.)
Pv_star = params(11); % Dimensionless vapor saturation pressure at the far field temperature
Req = params(12); % Dimensionless equilibrium bubble radius (LIC only) % JY!!!
%R1eq = params(13); R2eq = params(14); R3eq = params(15);
PA_star = params(13); % Dimensionless amplitude of the ultrasound pulse (UIC only)
omega_star = params(14); % Dimensionless frequency of the ultrasound pulse (UIC only)
delta_star = params(15); % Dimensionless time shift for the ultrasound pulse (UIC only)
n = params(16); % Exponent that shapes the ultrasound pulse (UIC only)

Ca = params(17); %Ca1 = params(21); Ca2 = params(22); Ca3 = params(23);
Re = params(18); %Re1 = params(25); Re2 = params(26); Re3 = params(27);
 
% dp1 = params(28); dp2 = params(29); dp3 = params(30);
% d12 = params(31); d13 = params(32); d23 = params(33);

% Se_f = 1/Ca/2*(5-4/lambda_Y-1/lambda_Y^4);

% Extract quantities from the state vector
R = x(1); % Bubble wall radius
U = x(2); % Bubble wall velocity
P = x(3); % Internal bubble pressure 
Se = x(4); 
Theta = x(5:(NT+4)); % Variable relating to internal temp (theta)
k = x((NT + 5):(2*NT+4)); % Vapor mass fraction (k) 


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
Tp = (A_star - 1 + sqrt(1+2.*A_star.*Theta))./A_star; % Dimensionless temperature T/T_inf
Kp_star = A_star.*Tp+B_star; % Dimensionless mixture thermal conductivity field
Rpmix = k.*Rv + (1-k).*Ra; % Mixture gas constant field (J/kg-K)

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


Dk(end-3:end) = 0;
DDk(end-3:end) = 0;

%%******************************************

%******************************************
% Internal bubble pressure evolution equation
pdot = 3/R*(-kappa*P*U + (kappa-1)*chi*DTheta(end)/R ...
    + kappa*P*fom*Rv*Dk(end)/(R*Rpmix(end)*(1-k(end))));

%******************************************

%******************************************
% Dimensionless mixture velocity field inside the bubble
Upmix = ((kappa-1).*chi./R.*DTheta-R.*yk.*pdot./3)./(kappa.*P) + fom./R.*(Rv-Ra)./Rpmix.*Dk;
%******************************************

%******************************************
% Evolution equation for the temperature (theta) of the mixture inside the bubble
Theta_prime = (pdot + (DDTheta).*chi./R.^2).*(Kp_star.*Tp./P.*(kappa-1)./kappa) ...
    - DTheta.*(Upmix-yk.*U)./R ...
    + fom./(R.^2).*(Rv-Ra)./Rpmix.*Dk.*DTheta;
Theta_prime(end) = 0; % Dirichlet BC at the bubble wall

%******************************************

%******************************************
% Evolution equation for the vapor concentration inside the bubble
k_prime = fom./R.^2.*(DDk + Dk.*(-((Rv - Ra)./Rpmix).*Dk - DTheta./sqrt(1+2.*A_star.*Theta)./Tp)) ...
    - (Upmix-U.*yk)./R.*Dk;
k_prime(end) = 0; % Dirichlet BC at the bubble wall

%******************************************




%******************************************
% Elastic stress in the material
%     (viscous contribution is accounted for in the Keller-Miksis equation)
if strcmp(cav_type,'LIC') == 1

    % Sp = -1/2/Cap* ( 5 - (Rpeq/Rp)^4 - 4*Rpeq/Rp ) - 4/Rep*Up/Rp;
    % S1 = -1/2/Ca1* ( 5 - (R1eq/R1)^4 - 4*(R1eq/R1) ) - 4/Re1*U1/R1;
    % S2 = -1/2/Ca2* ( 5 - (R2eq/R2)^4 - 4*(R2eq/R2) ) - 4/Re2*U2/R2;
    % S3 = -1/2/Ca3* ( 5 - (R3eq/R3)^4 - 4*(R3eq/R3) ) - 4/Re3*U3/R3;

    alpha = 0; %JY!!!

    Rst1 = R/Req;
    Sedot = -2*U/R*(1-3*alpha)*(1/Rst1 + 1/Rst1^4)/Ca - ...
          2*alpha*U/R*(1/Rst1^8 + 1/Rst1^5 + 2/Rst1^2 + 2*Rst1)/(Ca); % + 4/Rep*Up^2/Rp^2; 


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
rdot = U;
udot = ((1+U/C_star)*(P - 1/(We*R) + Se - 4*U/(Re*R) - 1 - Pext)  ...
        + R/C_star*(pdot + U/(We*R^2) + Sedot + 4*U^2/(Re*R^2) - Pextdot) ...
        - (3/2)*(1-U/(3*C_star))*U^2)/((1-U/C_star)*R + 4/(C_star*Re));

% udot = ( (P - 1/(We*R) + S - 1 - Pext) - (3/2)*U^2 ) / ( R );
% ============================================
% Keller-Miksis equations
% rdot = U;
% udot = ((1+U/C_star)*(P - 1/(We*R) + Se - 4*U/(Re*R) - 1 - Pext)  ...
%     + R/C_star*(pdot + U/(We*R^2) + Sedot + 4*U^2/(Re*R^2) - Pextdot) ...
%     - (3/2)*(1-U/(3*C_star))*U^2)/((1-U/C_star)*R + 4/(C_star*Re));
%******************************************

dxdt = [rdot; ...
    udot;... 
    pdot; ...
    Sedot; ...
    Theta_prime; ...
    k_prime];

end