function [r, drdS] = prod_starch_glc_mal()
% enzyme specific production rate inhibited by glcuose and maltose
% Yoo et al., 1988. Biotech Bioeng
Ki_g = 35.1751;  % mM
n_g = 4.255;  % unitless

Ki_m = 22.5066; % mM
n_m = 5.222; % unitless

Ka_starch = 1;  % g/L


% Km / [S]. S should just be the amino acids and E is the ribosome
% value unknown. Just treated as a scaling factor for Vmax. Using 1 is fine
KmOverS = 1;

% Vmax here is kcat multiplied by ribosome concentration g/hr/gDW ~40 unit/mgDW/hr


    % Catalytic Hill's inhibition and activation
r = @(starch, glc, mal, Vmax) Vmax ./ (KmOverS + Ka_starch / starch + (glc / Ki_g) .^ n_g + (mal / Ki_m) .^ n_m + 1);

% [drdStarch, drdGlucose, drdMaltose]
drdS = @(starch, glc, mal, Vmax) (Vmax / ((KmOverS + (glc / Ki_g) .^ n_g + (mal / Ki_m) .^ n_m + 1) * starch + Ka_starch) ^ 2) ...
* [Ka_starch, ...
starch ^ 2 * n_g * glc ^ (n_g - 1) / (Ki_g ^ n_g), ...
starch ^ 2 * n_m * mal ^ (n_m - 1) / (Ki_m ^ n_m)];

end
% beta_starch = 2;  % arbitrary activation factor
% Hill's inhibition and non-essential activation
% r = KmOverS * (1 + starch / Ka_starch + (glc / Ki_g) .^ n_g + (mal / Ki_m) .^ n_m) ./ (1 + beta_starch * starch / Ka_starch) + 1;
% r = Vmax ./ r;
