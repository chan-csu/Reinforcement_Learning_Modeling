% saveDir = 'chemo_ode_cplex_fwdEuler_20190510';
% % Dtest = [0.5];%[0.02, 0.1 0.2 0.5 0.8];
% % testAMYP = [0.01 0.02 0.05 0.08 0.1 0.12 0.15 0.2 0.3 0.5 0.8 1];
% % testAMYP = 0.05;
% testAMYP  = [0.01 0.03 0.1 0.3] ;
% Dtest = [0.05 0.1 0.3 0.6 1];

% amylase production rate x dilution rate to plot
testAMYP  = [0.01 0.03 0.1 0.3] ;
Dtest = [0.05 0.1 0.3 0.6 1];%[0.02, 0.1 0.2 0.5 0.8];

saveDir = 'chemo_ode_cplex_fwdEuler_20190510';

m = 2;
n = 5;

nSim = 1:numel(testAMYP);
ylab = {'Biomass (gDW/L)', 'Starch (g/L)', 'Glucose (mM)', 'Amylase (g/L)', 'Degraded amylase (g/L)'; ...
    'Glucose uptake (mmol/gDW/hr)', 'Amylase flux (mmol/gDW/hr)', 'Amylase production (g/gDW/hr)', 'Amylase degradtion (g/gDW/hr)', ''};

load([saveDir filesep 'pre.mat']);
for iD = 1:numel(Dtest)
    figure('Position', [107 200 1535 366]);
    
    for i = 1:numel(testAMYP)
        ct = 0;
        load(sprintf('%s%sd%.2famyp%.2f.mat', saveDir, filesep, Dtest(iD), testAMYP(i)));
        %%
        if size(Ct, 1) ~= numel(t_vect)
            Ct = Ct';
            fluxKineticsI = fluxKineticsI';
        end
        varPlot = [{xt, Ct(:, ind.s), Ct(:, ind.glc), Ct(:, ind.amy), Ct(:, ind.amyd)}; ...
            mat2cell(fluxKineticsI(:, [4 1:3]), size(fluxKineticsI, 1), ones(4, 1)), {[]}];
        for j = 1:m
            for k = 1:n
                ct = ct + 1;
                if i == 1
                    ax(j, k) = subplot(m, n, ct);
                    hold on
                    xlabel('Time (hours)')
                    ylabel(ylab{j, k})
                end
                if ~(j == m && k == n)
                    plot(ax(j, k), t_vect, varPlot{j, k});
                end
                
                if j == 1 && k == 1
                    title(sprintf('D = %.2f h^{-1}', Dtest(iD)));
                end
            end
        end
    end
end
