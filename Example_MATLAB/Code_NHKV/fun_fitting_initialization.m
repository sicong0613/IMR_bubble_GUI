function [struct_vars] = fun_fitting_initialization(varargin)
    % Provide the initial guess of the parameters to fit

    if nargin == 0

        struct_vars = [
            fun_P("U0",             100,         80,      120,  "log", "expansion1")
            fun_P("G",            8.0e+06,      1e6,    2e7,  "log", "expansion1")
            fun_P("mu",             0.226,      0.01,   1e2,  "log", "expansion2")
        ];

    elseif nargin == 1
        if ischar(varargin{1}) || isstring(varargin{1})
            load(varargin{1});
            struct_vars = struct_best_fit;
        else
            error('Input must be a string.');
        end
    elseif nargin == 2
        if ischar(varargin{1}) || isstring(varargin{1})
            load(varargin{1});
            mat_name = varargin{1};
            var_num = varargin{2};
        else
            error('Input must be a string.');
        end
    end
end
