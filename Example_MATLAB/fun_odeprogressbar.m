function status = odeprogress_singlewindow(t, y, flag, tspan, maxSec)
    % Single-window progress UI (no nested dialogs) for odeXX solvers.
    % - One uifigure with a custom progress bar + centered label + ETA + Cancel
    % - Call with: opts = odeset('OutputFcn', @(t,y,flag) odeprogress_singlewindow(t,y,flag,[t0 tf],[]));
    %              [t,y] = ode15s(@f, [t0 tf], y0, opts);
    
    persistent S
    status = false;
    
    switch flag
        case 'init'
            if numel(tspan) < 2, error('tspan must be [t0 tf]'); end
            t0 = tspan(1); tf = tspan(end);
            S.tmin = min(t0,tf); S.tmax = max(t0,tf);
            S.L    = S.tmax - S.tmin + eps;
    
            % timers + ETA smoothing
            S.tStart     = tic;
            S.lastUpdate = tic;
            S.maxSec     = []; if nargin>=5 && ~isempty(maxSec), S.maxSec = maxSec; end
            S.p_smooth   = 0; S.alpha = 0.2; S.lastETA = NaN;
    
            % --- Build one window with grid layout ---
            W = 700; H = 160;
            S.fig = uifigure('Name','ODE progress', 'Position', centerOnScreen(W,H), ...
                             'Resize','off');
            gl = uigridlayout(S.fig,[3 4]);
            gl.RowHeight    = {40, 30, '1x'};
            gl.ColumnWidth  = {'1x','fit','fit',80};
    
            % Title / info label
            S.lbl = uilabel(gl, 'Text', sprintf('t* in [%.3g, %.3g]', S.tmin, S.tmax), ...
                            'HorizontalAlignment','center', 'WordWrap','on', ...
                            'FontName','Consolas', 'FontSize', 12);
            S.lbl.Layout.Row = 1; S.lbl.Layout.Column = [1 4];
    
            % Progress bar: background + fill panels
            S.pb_bg   = uipanel(gl, 'BackgroundColor',[0.90 0.90 0.90], 'BorderType','line');
            S.pb_bg.Layout.Row = 2; S.pb_bg.Layout.Column = [1 4];
            S.pb_fill = uipanel(S.pb_bg, 'BackgroundColor',[0.00 0.45 0.74], 'BorderType','none');
            S.pb_fill.Position = [1 1 1 S.pb_bg.Position(4)];  % width updated later
    
            % Status text (elapsed, ETA)
            S.msg = uilabel(gl, 'Text', '0.0%  |  elapsed 00:00:00  |  ETA --:--:--', ...
                            'HorizontalAlignment','center', 'FontName','Consolas');
            S.msg.Layout.Row = 3; S.msg.Layout.Column = [1 3];
    
            % Cancel button and window-close hook
            S.btn = uibutton(gl, 'Text', 'Cancel', 'ButtonPushedFcn', @onCancel);
            S.btn.Layout.Row = 3; S.btn.Layout.Column = 4;
            S.fig.CloseRequestFcn = @onClose;   % clicking [X] also requests stop
    
        case ''   % normal step
            if isempty(S) || ~isvalid(S.fig)
                status = true; return;
            end
            % Stop if user requested
            if isappdata(S.fig,'user_stop') && getappdata(S.fig,'user_stop')
                status = true; delete(S.fig); return;
            end
            if isempty(t), return; end
    
            % Throttle UI updates (~10 Hz)
            if toc(S.lastUpdate) >= 0.1
                S.lastUpdate = tic;
    
                tt  = t(end);
                ttC = min(max(tt, S.tmin), S.tmax);
                p   = max(0, min(1, (ttC - S.tmin)/S.L));
    
                % Smooth progress for steadier ETA
                if S.p_smooth==0, S.p_smooth=p; else, S.p_smooth=(1-S.alpha)*S.p_smooth + S.alpha*p; end
                elapsed = toc(S.tStart);
    
                % ETA
                if S.p_smooth >= 0.01
                    eta = elapsed*(1 - S.p_smooth)/max(S.p_smooth, eps);
                    if isnan(S.lastETA), S.lastETA=eta; else, S.lastETA=0.7*S.lastETA + 0.3*eta; end
                    etaStr = sec2hms(S.lastETA);
                else
                    etaStr = '--:--:--';
                end
    
                % Update bar width
                try
                    bgpos  = S.pb_bg.Position;
                    fracW  = max(0,min(1,p));
                    S.pb_fill.Position = [1 1 max(1, fracW*(bgpos(3)-2)) bgpos(4)];
                catch
                    % layout not ready yet; skip
                end
    
                % Update text
                S.lbl.Text = sprintf('t* = %.6g  /  [%.3g, %.3g]', tt, S.tmin, S.tmax);
                S.msg.Text = sprintf('%5.1f%%  |  elapsed %s  |  ETA %s', 100*p, sec2hms(elapsed), etaStr);
    
                drawnow limitrate;
            end
    
            % Optional stops
            if ~isempty(S.maxSec) && toc(S.tStart) > S.maxSec, status = true; return; end
            if any(~isfinite(y(:))), status = true; return; end
    
        case 'done'
            if ~isempty(S) && isfield(S,'fig') && isvalid(S.fig), delete(S.fig); end
            S = [];
end

% ===== local helper functions (callbacks & utilities) =====
function onCancel(src, ~)
    % Button callback: mark stop on the parent figure (src is the button).
    fig = ancestor(src,'figure');              % find enclosing uifigure
    if ~isempty(fig) && isvalid(fig)
        setappdata(fig,'user_stop',true);      % flag checked in the OutputFcn
    end
    end
    
    function onClose(src, ~)
    % Window close [X]: mark stop then let OutputFcn delete the figure.
    setappdata(src,'user_stop',true);
    end
    
end  % <-- close primary function

function pos = centerOnScreen(W,H)
    ss = get(0,'ScreenSize');
    pos = [ss(1)+0.5*(ss(3)-W), ss(2)+0.6*(ss(4)-H), W, H];
end

function s = sec2hms(x)
    if ~isfinite(x) || x<0, s='--:--:--'; return; end
    x = round(x); hh=floor(x/3600); mm=floor(mod(x,3600)/60); ss=mod(x,60);
    s = sprintf('%02d:%02d:%02d',hh,mm,ss);
end
