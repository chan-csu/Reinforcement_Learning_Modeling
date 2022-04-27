function [x, y, varargout] = odeFwdEuler(fun, xSpan, y0, dx, nNonStateVar)
% Simple forward Euler integrator
%
% [x, y, v_1, v_2, ..., v_nNonStateVar] = odeFwdEuler(fun, xSpan, y0, dx, nNonStateVar)
%
% INPUTS:
%    fun:          function handle for the differential equation to be integrated. See e.g., ode45,
%                  with the output arguments being [dydx, v_1, v_2, ..., v_nNonStateVar] (see below)
%    xSpan:        range for x over which the ode is integrated, e.g., [0, 1]
%    y0:           initial y value
%    dx:           step size for change in x
%    nNonStateVar: number of non-state variables to return
%
% OUTPUTS:
%    x:            vector of values of x for which integration is performed, equal to xSpan(1):dx:xSpan(2)
%    y:            the corresponding values for y
%    v_1,...,v_nNonStateVar: additional non-state variables to return

if nargin < 4 || isempty(nNonStateVar)
    nNonStateVar = 0;
end

if numel(xSpan) == 1
    xSpan = [0 xSpan];
end

% pre-assign x, y
x = xSpan(1):dx:xSpan(end);
y = zeros(numel(y0), numel(x));
y(:, 1) = y0(:);

nonStateVar = cell(nNonStateVar, numel(x));

for j = 1:(numel(x) - 1)
    % compute dy/dx and the additional output arguments
    [dy, nonStateVar{:, j}] = fun(x(j), y(:, j));
    % get the next y
    y(:, j + 1) = y(:, j) + dy(:) * dx;
end
[dy, nonStateVar{:, end}] = fun(x(end), y(:, end));

% compute the end point if not yet
if x(end) < xSpan(end)
    x(end + 1) = xSpan(end);
    y(:, end + 1) = y(:, end) + dy * (x(end) - x(end - 1));
    [~, nonStateVar{:, end + 1}] = fun(x(end), y(:, end));
end

nonStateVar = cellfun(@(x) x(:), nonStateVar, 'UniformOutput', false);
varargout = cell(1, nNonStateVar);
for j = 1:nNonStateVar
    varargout{1, j} = nonStateVar(j, :);
end
end
