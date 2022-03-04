% This is the copied script for the heterogenous mix of E. coli (amylase
% and non-amylase producing)
% This script simulates the growth of a homogeneous population of E. coli
% secreting extracellular amylase to break down starch into glucose for carbon source.
% It assumes a chemostat bioreactor where fresh medium is continuously drained in
% and the culture is continuously drained out.

%% some simulation options
options = struct();
% simulation time (hour)
options.tFinal = 1;
% size of each time step
dt = 1/600;

% amylase production rate to test
testAMYP  = 0.01;%[0.01 0.03 0.1 0.3] ;
% dilution rate to test
Dtest = 0.05;%[0.05 0.1 0.3 0.6 1];%[0.02, 0.1 0.2 0.5 0.8];
VmaxVector = [0.05 0];

% directory for saving the data
saveDir = [pwd filesep 'test'];%'/Users/macbookpro/Documents/chemo_ode_cplex_fwdEuler_20190709';

%% setup the base model and mutant models

% read the model
model = readCbModel('iJO1366.mat');

% formatting:
% change _e to [e] etc.
model.mets = regexprep(model.mets, '_([a-zA-Z0-9])$', '\[$1\]');

% change EX_met_e to EX_met(e)
model.rxns = regexprep(model.rxns, '^EX_(\w+)_e$', 'EX_$1\(e\)');

% use the WT biomass reaction as objective function
model = changeObjective(model, 'BIOMASS_Ec_iJO1366_WT_53p95M',  1);

%%%%% now add one biomass metabolite for each mutant as done below
% % metabolite ID for the biomass
% biomassMetId = 'biomass_wt[e]';
%
% % Check if there is a biomass (pseudo-) metabolite in the biomass reaction for consistent representation.
% % Add if it is not there
% model = addMetabolite(model, biomassMetId, 'metName', 'Biomass WT');
% r = findRxnIDs(model, 'BIOMASS_Ec_iJO1366_WT_53p95M');
% m = findMetIDs(model, biomassMetId);
%
% model.S(m, r) = 1;
%
% % add an export reaction for the biomass metabolite
% model = addReaction(model, ['EX_' strrep(biomassMetId, '[e]', '(e)')], ...
%     'reactionFormula', [biomassMetId ' ->'], 'reactionName', 'Export reaction for biomass WT');

% Hydrolysis catalyzed by amylase. Assume an average chain length of 20
% (Wang, Ya-Jane & Wang, Linfeng. (2003). Physiochemical properties of common and waxy corn starches oxidized by different levels of sodium hypochlorite. Carbohydrate Polymers. 52. 207-217. 10.1016/S0144-8617(02)003041.)
% And assume amylase simply turns starch into glucose.
amylaseRxn = 'starch[e] + 19 h2o[e] -> 20 glc__D[e]';

% add the amylase production module to the model
nOrg = numel(VmaxVector);
% xc_model = model;
[model, xc_model, metAbbrs, rxnAbbrs] = addAmylaseToModel(model);


for i = 1:nOrg
    % add a biomass metabolite for each mutant in the template model
    options.biomassMetId{i} = ['biomass_org' num2str(i) '[e]'];
    options.biomassRxnId{i} = ['EX_biomass_org' num2str(i) '(e)'];
    model = addMetabolite(model, ['biomass_org' num2str(i) '[e]']);
    % add an exchange reaction for each biomass metabolite as well
    model = addReaction(model, ['EX_biomass_org' num2str(i) '(e)'], 'reactionFormula', ['biomass_org' num2str(i) '[e] ->']);
end

% get the formulae for the starch and biomass, which are added above
model = computeMetFormulae(model, 'knownMets', model.mets(~cellfun(@isempty, model.metFormulas())));

% create mutant models
models = repmat({model}, nOrg, 1);
for i = 1:nOrg
    % modify each mutant model to connect to the individual biomass metabolite
    rxnBiomass = findRxnIDs(models{i}, 'BIOMASS_Ec_iJO1366_WT_53p95M');
    metBiomass = findMetIDs(models{i}, ['biomass_org' num2str(i) '[e]']);
    models{i}.S(metBiomass, rxnBiomass) = 1;
    for i2 = 1:nOrg
        if i2 ~= i
            % shut down the exchange reaction for other biomass exchange reactions
            models{i} = changeRxnBounds(models{i}, ['EX_biomass_org' num2str(i2) '(e)'], 0, 'b');
        end
    end
end

% do FBA
%s = optimizeCbModel(model);

% look at the exchange fluxes, amylase export is among them
%printFluxVector(model, s.x, 1, 1);

% logical IDs for exchange reactions
rxnExId = sum(model.S ~= 0, 1) == 1;
xc_rxnExId = sum(xc_model.S ~= 0, 1) == 1;
% indices for the corresponding extracellular metabolites
[metExId, ~] = find(model.S(:, rxnExId));
[xc_metExId, ~] = find(xc_model.S(:, xc_rxnExId));

% [model.mets(metExId), model.rxns(rxnExId)]


%% setup the initial medium concentration vector

% number of extracellular metabolites in the microbial model
nC = numel(metExId);
% metabolite in the conc vector
metC = union(model.mets(metExId), xc_model.mets(xc_metExId), 'stable');
[~, ind_ic] = ismember(model.mets(metExId), metC);
[~, ind_xc] = ismember(xc_model.mets(xc_metExId), metC);

% indices for some important metabolites/biomass/proteins among all the extracellular metabolites
ind = struct();
ind.o2 = strcmp(metC, 'o2[e]');  % oxygen
ind.ac= strcmp(metC, 'ac[e]');  % acetate
ind.glc = strcmp(metC, 'glc__D[e]');  % glucose
%ind.amy = strcmp(model.mets(metExId), metAbbrs.amylase);  % amylase
%ind.amyd = strcmp(model.mets(metExId), metAbbrs.amylase_d);  % degraded amylase
%ind.s = strcmp(model.mets(metExId), metAbbrs.starch);  % starch
[~, ind.x] = ismember(options.biomassMetId, metC);  % biomass

ind.s = strcmp(metC(xc_metExId), 'starch[e]');  % starch
ind.amy = strcmp(metC(xc_metExId), 'amylase[e]');  % amylase
ind.amyd = strcmp(metC(xc_metExId), 'amylase_d[e]');  % degraded amylase


% % number of additional ex mets in xc_model
% met_xc = setdiff(xc_model.mets, model.mets, 'stable');
% nC_xc = numel(met_xc);
% % index mapping for microbial model
% ind_ic = 1:nC;
% ind_xc = nC + 1 : nC + nC_xc;
% initial concentration vector
C0 = zeros(numel(metC), 1);
% use the default minimal medium from the model
% '1000' m virtually non-limiting substrates
C0(ind_ic(model.lb(rxnExId) < 0)) = 1000;
% some initial glucose for feasibility solution (satisfy ATP maintenance and growth)
C0(ind.glc) = 0.01;
% oxygen supply non-limiting
C0(ind.o2) = 1000;
% initial biomass
C0(ind.x) = 0.005;



% Assume volume = 1 L

% 1 g/L starch available
% get the molecular weight of the starch in the model
starchMw = getMolecularMass(model.metFormulas(findMetIDs(xc_model, metAbbrs.starch)), 0, 1);
C0(ind.s) = 1 / starchMw * 1000; % 1 (g/L) / (g/mol) * (mmol/mol) = mM

% finite oxygen uptake by the organism
model = changeRxnBounds(model, 'EX_o2(e)', -18.5, 'l');
% non-limiting cobalamine I uptake
model = changeRxnBounds(model, 'EX_cbl1(e)', -1000, 'l');

% non-limiting substrates
% [all except glucose, O2, biomass (which is for export only actually), and starch]
nonlimitSub = false(numel(metC), 1);
nonlimitSub(ind_ic) = model.lb(rxnExId) < 0;
%nonlimitSub(ind.glc | ind.o2 | ind.s) = false;
nonlimitSub(ind.glc | ind.o2) = false;
nonlimitSub(ind.s) = false;
nonlimitSub(ind.x) = false;

% open all other uptake reactions except for sink reactions. The actual uptake bounds are determined by the concentration vector
% meanwhile keep the existing max. uptake rate (organism's uptake capacity)
model.lb(rxnExId(:) & model.lb >= 0 & ~strncmp(model.rxns, 'DM_', 3)) = -1000;

% print the exchange reactions, metabolites and their initial concentration
%writeCell2Text(disp([{'rxns', 'mets', 'init. conc.'} model.rxns(rxnExId), model.mets(metExId), num2cell(C0)]));
%writeCell2Text(disp([{'rxns', 'mets', 'init. conc.'} xc_model.rxns(rxnExId), xc_model.mets(metExId), num2cell(C0)]));


%% kinetics

%%%%% kinetic constants
% get the elemental composition of the starch in the model
% Added by dr. chan
xc_model.metFormulas(findMetIDs(xc_model, 'starch[e]')) = {'C12H24O12'};
[Ematrix, elements] = computeElementalMatrix(xc_model, metAbbrs.starch, 0, 1);
% index for element C
eleCarbon = strcmp(elements, 'C');

% alpha-amylase kinetic parameters
% https://www.brenda-enzymes.org/enzyme.php?ecno=3.2.1.1&Suchword=subtilis&reference=&UniProtAcc=&organism%5B%5D=Bacillus+subtilis&show_tm=0
% K_m = 1 g/L for starch of any length. Convert into mM
Km_amy = 0.1 / starchMw * 1000;
% k_cat converted into mmol starch / hr / gram protein
kcat_amy = 4133 / 1e3 * 6 ./ Ematrix(1, eleCarbon) * 60 * 1000;
% 4133 umol gluose/min/mg protein * (mmol/umol) * (C-mmol/mmol glucose) * (mmol starch/C-mmol) * (min/hr) * (g protein / mg protein)
% 12399 mmol starch (C120) / hr / g protein

% protein degradation time constant
K_deg_amyl = 0.5771;

% V_max for glucose uptake (from BioNumbers)
Vmax_glc = 10;
% K_m for glucose uptake (from BioNumbers)
Km_glc = 0.00610582;
%%%%%

%%%%% kinetics info true for all mutants
% reactions that have kinetics associated
kineticsRxns = {rxnAbbrs.amylaseProduction; rxnAbbrs.glucoseExchange};
kineticsRxns_xc = {rxnAbbrs.amylase; rxnAbbrs.amylaseDegradation};
% Use the kinetic equations as upper bound (u), lower bound (l), or both (b)
kineticsBounds = {'b', 'b'};
kineticsBounds_xc = {'b', 'b'};

% kinetics for amylase production
% C: concentration vector,
% Vmax_amyp: V_max for amylase production, as a mutant-dependent parameter
% prod_starch_glc_ma is separate function for the kinetic equation
r = prod_starch_glc_mal;
kineticsAmylaseProd = @(C, Vmax_amyp) r(C(ind.s), C(ind.glc), 0, Vmax_amyp);  % assume no maltose at all. Pure starch-to-glucose conversion
% idX: index for a mutant's biomass metabolite among the extracellular mets
kineticsAmylaseDegrad = @(C) K_deg_amyl * C(ind.amy);

% uptake reactions that use generic kinetics: any limiting substrates that
% are not glucose, starch, biomass or amylase
kineticsGenericRxns = model.rxns(rxnExId);
%kineticsGenericRxns = kineticsGenericRxns(C0 < 1000 & model.lb(rxnExId) < 0 & ...
    %~ismember(kineticsGenericRxns, [{'EX_glc__D(u)';'EX_starch(e)'; 'EX_amylase(e)'; 'EX_amylase_d(e)'; 'EX_h2o(e)'}; options.biomassRxnId(:)]));
kineticsGenericRxns = kineticsGenericRxns(C0(ind_ic) < 1000 & model.lb(rxnExId) < 0 & ...
    ~ismember(kineticsGenericRxns, [{'EX_amylase(e)'}; options.biomassRxnId(:)]));
% use generic kinetics for lower bounds only
% (limit the max uptake, but not imposing forced uptake)
kineticsGenericBounds = repmat({'l'}, 1, numel(kineticsGenericRxns));

% the indices of the reactions with generic kinetics among all exchange rxns
[~, id] = ismember(kineticsGenericRxns, model.rxns(rxnExId));
kineticsGenericEqs = cell(numel(kineticsGenericRxns), 1);
% default lower bounds for exchange reactions (i.e.,  max. uptake rates)
lbEx0 = model.lb(rxnExId);
%%%%%

%%%%% assign kinetic equations to individual mutants
% index for amylase production among the reactions with kinetics
%amypInKinetis = find(strcmp({kinetics.rxn}, 'AMYLp'));
options.kinetics = cell(nOrg, 1);

for i =1:nOrg
    % kinetic equations corresponding to kineticsRxns
    % (all kinetic equations are function handle of C)
    kineticsEqs = { ...%@(C) kcat_amy * C(ind.amy) / (Km_amy / C(ind.s) + 1); ...  % amylase reaction
        @(C) kineticsAmylaseProd(C, VmaxVector(i)); ...  % amylase enzyme production, take Vmax_amyp = 0 for now. Changed in each iteration below
        ... % @(C) kineticsAmylaseDegrad(C, ind.x(i)); ...  % amylase degradation
        @(C) -Vmax_glc / (Km_glc / C(ind.glc) + 1)};  % glucose uptake

    for j = 1:numel(kineticsGenericRxns)
        % use generic uptake kinetics to avoid stiffness in integration
        % Monod / Michaelis-Menten kinetics:
        % kineticsGenericEqs{j} = @(C) -200 * C(id(j)) / (C(id(j)) + 0.1);

        % max of organism uptake capacity and substrate availability (may cause stiff integration)
        kineticsGenericEqs{j} = @(C) max(lbEx0(id(j)), -C(ind_ic(id(j))) / C(ind.x(i)) / dt);
    end

    options.kinetics{i} = struct('rxn', [kineticsRxns; kineticsGenericRxns], ...
    'eqs', [kineticsEqs; kineticsGenericEqs], 'type', [kineticsBounds, kineticsGenericBounds]');
end

kineticsEqs_xc = {@(C) kcat_amy * C(ind.amy) / (Km_amy / C(ind.s) + 1); ...  % amylase reaction
        kineticsAmylaseDegrad; ...  % amylase degradation
        };
options.kinetics_xc = struct('rxn', kineticsRxns_xc, ...
    'eqs', kineticsEqs_xc, 'type', kineticsBounds_xc(:));

% stopping criterion for the simulation: if relative changes of all limiting substrates < 0.0001
% stoppingCriterion = @(t, C, dC) (C(ind.s) < 1e-9 & C(ind.glc) < 1e-9) | t >= 50;
stoppingCriterion = @(t, C, dC) max(abs(dC(~nonlimitSub(:) & C(:) ~= 0))./C(~nonlimitSub(:) & C(:) ~= 0)) < 1e-4;

% simulation options to be passed to the dFBA function
options.C0 = C0;
options.dt = dt;
options.stoppingCriterion = stoppingCriterion;
options.Cfeed = C0;
% no biomass in the feed stream
options.Cfeed(ind.x) = 0;

% parameters for Cplex (an optimization solver)
param = struct();
[param.simplex.display, param.tune.display, param.barrier.display,...
    param.sifting.display, param.conflict.display, param.mip.tolerances.integrality] = deal(0);
[param.simplex.tolerances.optimality, param.simplex.tolerances.feasibility] = deal(1e-9,1e-8);
param.read.scale = -1;
param.clonelog = 0;
options.cplexParam = param;
options.cplexParam.barrier.convergetol = 1e-9;
options.cplexParam.qpmethod = 6;

% flux variables to be outputted
options.fluxFBAName = 'fluxFBAi';
options.fluxKineticsName = 'fluxKineticsI';

options.ind_ic = ind_ic;
options.ind_xc = ind_xc;
options.xc_rxnExId = xc_rxnExId;
%options.kinetics_xc = kinetic_xc;
%% simulation

% create the directory for saving data if not existing already
if ~exist(saveDir, 'dir')
    mkdir(saveDir)
end

% save the preparatory data
save([saveDir filesep 'pre.mat'], 'ind', 'options','model','Dtest','saveDir','testAMYP')




for iD = 1:numel(Dtest)
    % test different dilution rates
    options.dilutionRate = Dtest(iD);
    fprintf('Start integration  %04d-%02d-%02d %02d:%02d:%02.0f\n', clock)

        % still testing different amylase prod rates, but in different
        % models this time. (kinEqs -> models)
%    for i = 1 : numel(kinEqs)
%        options.kinetics(amypInKinetis).eqs = @(C) kineticsAmylaseProd(C, kinEqs(i));
        [t, xt, Ct, fluxFBAi, fluxFBA_xc, fluxKineticsI] = dFBA_kinetics_chemo_ode_cplex_fwdEuler(models, xc_model, options);
    %%for i = 1 : numel(testAMYP)
        % test different amylase production rates
        %%options.kinetics(amypInKinetis).eqs = @(C) kineticsAmylaseProd(C, testAMYP(i));

        % run dynamic FBA (dFBA)
        % dFBA_kinetics_chemo_ode_cplex_fwdEuler is a dFBA function for
        % simulating chemostat using the solver Cplex with the simple
        % forward Euler integration method
        %%[t, xt, Ct, fluxFBAi, fluxKineticsI] = dFBA_kinetics_chemo_ode_cplex_fwdEuler(model, options);
        % t: time vector
        % xt: biomass over time
        % Ct: concentration vector over time
        % fluxFBAi: flux vectors over time
        % fluxKineticsI: flux vectors for reactions with kinetics only

        % save the results
        save(sprintf('%s%sd%.2famyp%.2f.mat', saveDir, filesep, Dtest(iD), testAMYP(i)), ...
            't', 'xt', 'Ct', 'i', 'iD', 'testAMYP', 'Dtest', 'options', ...
            'fluxFBAi', 'fluxKineticsI', 'ind', '-v7.3');

        fprintf('%3d/%3d  AMYP: %.4f  max X: %.4f  %04d-%02d-%02d %02d:%02d:%02.0f\n', i, numel(testAMYP), testAMYP(i), max(xt), clock)
end
    if 0 %change to 'if 1' for the plot
        %% PLOT

        X = 1; %edit
        scale = 'log';
        timePoint = 1:15:size(X, 1);
        figure('Position', [107 500 1500 300]);
        ax(1) = subplot(1, 5, 1);
        plot(testAMYP, X(timePoint, :));
        ax(1).XScale = scale;
        ylabel('X (g/L)')
        ax(2) = subplot(1, 5, 2);
        plot(testAMYP, S(timePoint, :));
        ylabel('Starch (mM)')
        ax(2).XScale = scale;
        ax(3) = subplot(1, 5, 3);
        plot(testAMYP, G(timePoint, :));
        ylabel('glucose (mM)')
        ax(3).XScale = scale;
        ax(4) = subplot(1, 5, 4);
        plot(testAMYP, A(timePoint, :));
        ylabel('Amylase (g/L)')
        ax(4).XScale = scale;
        ax(5) = subplot(1, 5, 5);
        plot(testAMYP, Ad(timePoint, :));
        ylabel('Degraded amylase (g/L)')
        ax(5).XScale = scale;
    end

    if 0
        %%
        c = clock;
        save(sprintf('simRes_%04d%02d%02d_D%.2f.mat', c(1:3), Dtest(iD)));
    end
    if 0
        %% plotting biomass
        %plotting biomass as a function of time (AKA biomass is y, and time is x)
        %setting y as biomass, glucose, oxygene, and acetate by
        m = 2;
        n = 4;
        ct = 0;
        nSim = 1:numel(testAMYP);
        varPlot = [{X(:, nSim), S(:, nSim), G(:, nSim), A(:, nSim)}; ...
            permute(mat2cell(permute(fluxKinetics(:, [4, 1:3], nSim), [1 3 2]), ...
            size(fluxKinetics,1),size(fluxKinetics,3),ones(1,4)), [1 3 2])];
        ylab = {'Biomass (gDW/L)', 'Starch (g/L)', 'Glucose (mM)', 'Amylase (g/L)'; ...
            'Glucose uptake (mmol/gDW/hr)', 'Amylase flux (mmol/gDW/hr)', 'Amylase production (g/gDW/hr)', 'Amylase degradtion (g/gDW/hr)'};
        figure('Position', [107 200 1535 366]);
        for j = 1:m
            for k = 1:n

                ct = ct + 1;
                subplot(m, n, ct);
                plot(t_vect, varPlot{j, k});
                xlabel('Time (hours)')
                ylabel(ylab{j, k})
                if j == 1 && k == 1
                    title(sprintf('D = %.2f h^{-1}', Dtest(iD)));
                end
            end
        end

    end
