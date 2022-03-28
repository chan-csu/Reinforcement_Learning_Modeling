function [t, xt, Ct, fluxFBA, fluxFBA_xc, fluxKinetics] = dFBA_kinetics_chemo_ode_cplex_fwdEuler(models, xc_model, options)
% Perform dFBA using forward Euler method and Cplex as solver for a
% chemostat bioreactor
%
% USAGE:
%    [t, xt, Ct, fluxFBA, fluxKinetics] = dFBA_kinetics_chemo_ode_cplex_fwdEuler(model, options)
%
% INPUTS:
%    model:      COBRA model
%    options:    option structure with the following fields
%                *.biomassMetId:   ID for the biomass metabolite
%                *.C0:             initial concentration
%                *.dilutionRate:   dilution rate of the chemostat reactor
%                *.Cfeed:    concentration vector of the feed stream
%                *.kinetics: structure array for implementing kinetics.
%                            Each structure has the following field:
%                            *.rxn    ID for the reaction to be constrained by kinetics
%                            *.eqs    kinetic equation constraining the reaction
%                            *.type   'u', 'l', or 'b' to use the kinetics as upper
%                                     bound, lower bound or both
%                *.tFinal:   total simulation time (hour)
%
% OUTPUTS:
%    t:             time vector
%    xt:            biomass over time
%    Ct:            concentration vector over time
%    fluxFBAi:      flux vectors over time
%    fluxKineticsI: flux vectors for reactions with kinetics only

% retrieve the parameters
[biomassMetId, C0, dilutionRate, Cfeed, kinetics, tFinal, ind_ic, ind_xc, kinetics_xc] = ...
    deal(options.biomassMetId, options.C0, options.dilutionRate, options.Cfeed, ...
    options.kinetics, options.tFinal, options.ind_ic, options.ind_xc, options.kinetics_xc);

% solver parameter: feasibility tolerance
feasTol = getCobraSolverParams('LP', 'feasTol');
% define variables outside for loop
nOrg = numel(models);
%%xc_model_cell = repmat({xc_model}, nOrg, 1);
for i = 1 : nOrg
% logical IDs for exchange reactions
    rxnExId{i} = find(sum(models{i}.S ~= 0, 1) == 1);
    %%xc_rxnExId{i} = find(sum(xc_model_cell{i}.S ~= 0, 1) == 1);
    xc_rxnExId = find(sum(xc_model.S ~= 0, 1) == 1);
% indices for extracellular metabolites
    [metExId, ~] = find(models{i}.S(:, rxnExId{i}));
% the id of the biomass metabolite among all extracellular metabolites
    xInEx(i) = find(strcmp(models{i}.mets(metExId), biomassMetId{i}));
% take the reaction in the objective function as the biomass reaction
    rxnBmId = find(models{i}.c);
end

% get initial fluxes
% use Cplex
for i = 1 : nOrg
    lbEx0 = models{i}.lb(rxnExId{i});
    [m(i), n(i)] = size(models{i}.S);
    nEx_ic(i) = numel(rxnExId{i});
end

%combine the cells in 'models' before passing to solver (LP)

for i = 1 : nOrg
% create a Cplex object instance
    LP{i} = Cplex();
% display function not needed
    LP{i}.DisplayFunc = [];
% add variables, with biomass production as the objective

    LP{i}.addCols(models{i}.c, [], models{i}.lb, models{i}.ub);
    % add constraints (Sv = 0)

    LP{i}.addRows(zeros(m(i), 1), models{i}.S, zeros(m(i), 1));

% No uptake if the concentration is zero
    LP{i}.Model.lb(rxnExId{i}(C0(ind_ic) < 1e-7)) = 0;
end

% constraint the fluxes for reactions with kinetics
for i = 1 : nOrg
    for j = 1:numel(kinetics{i})
    % get the indices for the reactions with kinetics
        kinetics{i}(j).rxn = findRxnIDs(models{i}, kinetics{i}(j).rxn);
    end
% get the numerical fluxes for reactions with kinetics given the initial conc.
    fluxK = arrayfun(@(x) x.eqs(C0(ind_ic)), kinetics{i});
    fluxK(abs(fluxK) < feasTol) = 0;
% use the fluxes from kinetics as bounds for the optimization problem
    rxnChangeBounds = [kinetics{i}.rxn];

% change lower bounds
    LP{i}.Model.lb(rxnChangeBounds([kinetics{i}.type] == 'b' | [kinetics{i}.type] == 'l')) ...
        = fluxK([kinetics{i}.type] == 'b' | [kinetics{i}.type] == 'l');
% change upper bounds
    LP{i}.Model.ub(rxnChangeBounds([kinetics{i}.type] == 'b' | [kinetics{i}.type] == 'u')) ...
        = fluxK([kinetics{i}.type] == 'b' | [kinetics{i}.type] == 'u');
% set Cplex parameters
    LP{i} = setCplexParam(LP{i}, options.cplexParam);

% do FBA
    LP{i}.Model.sense = 'maximize';
    LP{i}.solve();

    % if no feasible solution, terminate
    if ~(checkSolFeas(LP{i}) < feasTol)
        warning('The model is infeasible')
        [t, xt, Ct, fluxFBA, fluxKinetics] = deal([]);
        return
    end
end

% LP solved for the extracellular model------------------------------
lbEx0_xc = xc_model.lb(xc_rxnExId);
[m_xc, n_xc] = size(xc_model.S);
nEx_xc = numel(xc_rxnExId);

LP_xc = Cplex();
LP_xc.DisplayFunc = [];
LP_xc.addCols(xc_model.c, [], xc_model.lb, xc_model.ub);
LP_xc.addRows(zeros(m_xc, 1), xc_model.S, zeros(m_xc, 1));

for j = 1:numel(kinetics_xc)
    kinetics_xc(j).rxn = findRxnIDs(xc_model, kinetics_xc(j).rxn);
end
fluxK = arrayfun(@(x) x.eqs(C0), kinetics_xc);
fluxK(abs(fluxK) < feasTol) = 0;
% use the fluxes from kinetics as bounds for the optimization problem
rxnChangeBounds = [kinetics_xc.rxn];
% change lower bounds
LP_xc.Model.lb(rxnChangeBounds([kinetics_xc.type] == 'b' | [kinetics_xc.type] == 'l')) ...
    = fluxK([kinetics_xc.type] == 'b' | [kinetics_xc.type] == 'l');
% change upper bounds
LP_xc.Model.ub(rxnChangeBounds([kinetics_xc.type] == 'b' | [kinetics_xc.type] == 'u')) ...
    = fluxK([kinetics_xc.type] == 'b' | [kinetics_xc.type] == 'u');
% set Cplex parameters
LP_xc = setCplexParam(LP_xc, options.cplexParam);
% do FBA
LP_xc.solve();
%----------------------------------------------------------------

% if no feasible solution, terminate
if ~(checkSolFeas(LP_xc) < feasTol)
    warning('The model is infeasible')
    [t, xt, Ct, fluxFBA_xc, fluxKinetics] = deal([]);
    return
end


for i = 1 : nOrg
% max biomass production
    maxBm{i} = LP{i}.Model.obj' * LP{i}.Solution.x;
% add a constraint to fix the biomass production to be maximum
    LP{i}.addRows(maxBm{i} - 1e-7, LP{i}.Model.obj', maxBm{i});
% minimize the exchange fluxes under maximum biomass production
    LP{i}.Model.obj(:) = 0;
% add variables for |v|
    LP{i}.addCols(ones(nEx_ic(i), 1), [],  zeros(nEx_ic(i), 1), max(abs(models{i}.lb(rxnExId{i})), abs(models{i}.ub(rxnExId{i}))));
% add constraint v - |v| < 0
    LP{i}.addRows(-inf(nEx_ic(i), 1), [sparse(1:nEx_ic(i), rxnExId{i}, 1, nEx_ic(i), n(i)), -speye(nEx_ic(i))], zeros(nEx_ic(i), 1));
% add constraint -v - |v| < 0
    LP{i}.addRows(-inf(nEx_ic(i), 1), [sparse(1:nEx_ic(i), rxnExId{i}, -1, nEx_ic(i), n(i)), -speye(nEx_ic(i))], zeros(nEx_ic(i), 1));
    LP{i}.Model.sense = 'minimize';
    LP{i}.solve();
% flux vector at max biomass production and min exchange fluxes
    v0{i} = LP{i}.Solution.x(rxnExId{i});

% remove the variables |v| (they are not needed anymore)
    LP{i}.delCols((n(i) + 1):(n(i) + nEx_ic(i)));
% remove the constraints v - |v| < 0 and -v - |v| < 0
    LP{i}.delRows((m(i) + 2):(m(i) + 1 + nEx_ic(i) * 2));
end
% store v0, the initial flux vector in the differential equation function
dFBA('init', 0, v0, n, rxnExId);
    %Loop over the altered dFBA function
    % dCdt = R * C(xInEx) + (Cfeed - C) * dilutionRate;

% odeOptions = odeset('OutputFcn', @(t, C, flag) dFBAoutput(t, C, flag, LP, lbEx0, dilutionRate, ...
%     Cfeed, kinetics, rxnBmId, rxnExId, xInEx, feasTol, fluxFBAName, fluxKineticsName), ...
%     'Events', @(t, C) stopButton(t, C, LP, lbEx0, dilutionRate, Cfeed, kinetics, rxnBmId, rxnExId, xInEx, feasTol), 'Stats', 'on');

% call the forward Euler differential equation integrator
% (ask 'odeFwdEuler' to integrate 'dFBA')

[t, Ct, fluxFBA, fluxKinetics] = odeFwdEuler(@(t, C) dFBA('compute', t, C, LP, LP_xc, lbEx0, dilutionRate, Cfeed, kinetics, kinetics_xc, rxnBmId, rxnExId, xc_rxnExId, xInEx, feasTol, ind_ic, ind_xc), ...
        [0 tFinal], C0, options.dt, 2);

% in case the dimension is not right
for i = 1 : nOrg
    if size(Ct, 1) == numel(t)
        xt = Ct(:, xInEx(:,i));
    else
        xt = Ct(xInEx(:,i), :);
    end
end
end

function [dCdt, fluxFBA, fluxFBA_xc, fluxKinetics] = dFBA(action, varargin)
% dynamic FBA, an FBA-embedded differential equation system
%
% USAGE:
%    1. dFBA('init', t, v0, n, rxnExId) to initialize some local data before integration
%    2. dCdt = dFBA('compute', t, C, LP, lbEx0, dilutionRate, Cfeed, kinetics, rxnBmId, rxnExId, xInEx, feasTol)
%       returns the vector of rate of change for the concentrations. Used
%       for integration
%
% INPUTS:
%    t:        time
%    v0:       initial flux vector
%    n:        number of reactions in the model
%    rxnExId:  indices for exchange reactions in the model
%    C:        concentration vector
%    LP:       Cplex solver instance
%    lbEx0:    the default lower bounds for exchange reactions
%    dilutionRate:   the dilution rate of the simulation
%    Cfeed:    concentration vector of the feed stream
%    kinetics: structure array for implementing kinetics. Each structure
%              has the following field:
%              *.rxn    ID for the reaction to be constrained by kinetics
%              *.eqs    kinetic equation constraining the reaction
%              *.type   'u', 'l', or 'b' to use the kinetics as upper
%                       bound, lower bound or both
%    rxnBmId:  ID for the biomass reaction
%    xInEx:    index for the biomass metabolite among the extracellular metabolites
%    feasTol:  feasibily tolerance (for rounding off zeros)
%
% OUPUTS:
%    dCdt:          the vector of rate of change of the extracellular concentrations
%    fluxFBA:       flux vector from FBA
%    fluxKinetics:  flux vector for reactions with kinetics, in the same
%                   order as the input kinetics,rxn

% (varargin is a special Matlab keyword for a variable length of input arguments)

% persistent variables for local use
persistent vEx
persistent t_v
persistent Q

[dCdt, fluxFBA, fluxKinetics] = deal([]);
    switch action
    case 'init'

        % preparatory step. Call before the actual integration to
        % initialize some local variables
        [t, v0, n, rxnExId] = deal(varargin{:});
        [Q, vEx] = deal(cell(numel(rxnExId), 1));
        nStore = 10;
        t_v = [t - 1; NaN(nStore - 1, 1)];
        for i = 1 : numel(v0)
            vEx{i} = zeros(nStore, numel(rxnExId{i}));
            vEx{i}(1, :) = v0{i}(:)';
            Q{i} = sparse(rxnExId{i}, rxnExId{i}, 1, n(i), n(i));
        end
    case 'compute'
        [t, C, LP, LP_xc, lbEx0, dilutionRate, Cfeed, kinetics, kinetics_xc, rxnBmId, rxnExId, xc_rxnExId, xInEx, feasTol, ind_ic, ind_xc] = deal(varargin{:});

        % solve the bacteria model
        for i = 1 : numel(LP)
        % max biomass
            LP{i}.Model.sense = 'maximize';
            LP{i}.Model.obj(:) = 0;
            LP{i}.Model.obj(rxnBmId) = 1;
        % no uptake is possible if concentration is 0
        %%%% This is a switch setting. May make the ode very stiff
            LP{i}.Model.lb(rxnExId{i}) = lbEx0; %(took this out 11/22)
            LP{i}.Model.lb(rxnExId{i}(C(ind_ic) < feasTol)) = 0;
        % impose fluxes constrained by kinetics
            fluxKinetics{i} = arrayfun(@(x) x.eqs(C), kinetics{i});
            fluxKinetics{i}(abs(fluxKinetics{i}) < feasTol) = 0;
            fluxKinetics{i} = fluxKinetics{i}(:);
            rxnChangeBounds = [kinetics{i}.rxn];
            LP{i}.Model.lb(rxnChangeBounds([kinetics{i}.type] == 'b' | [kinetics{i}.type] == 'l')) ...
                = fluxKinetics{i}([kinetics{i}.type] == 'b' | [kinetics{i}.type] == 'l');
            LP{i}.Model.ub(rxnChangeBounds([kinetics{i}.type] == 'b' | [kinetics{i}.type] == 'u')) ...
                = fluxKinetics{i}([kinetics{i}.type] == 'b' | [kinetics{i}.type] == 'u');

        % relax the maximum biomass constraint
            LP{i}.Model.lhs(end) = 0;
            LP{i}.Model.rhs(end) = 1000;
            LP{i}.solve();

        % choose a solution with a consumption/production profile most
        % similar to the previous time step.
            if LP{i}.Solution.status == 1
            % fix biomass production
                maxBm = LP{i}.Model.obj' * LP{i}.Solution.x;
                LP{i}.Model.lhs(end) = maxBm - 1e-7;
                LP{i}.Model.rhs(end) = maxBm;
            % change objective to min sum (v - v_prev)^2
                LP{i}.Model.obj(:) = 0;
            % choose the closest previous time point
                vEx{i}(t_v >= t, :) = 0;
                t_v(t_v >= t) = 0;
                [~, id] = min(t - t_v);
            % linear term in the objective - sum(v_prev * v)
                LP{i}.Model.obj(rxnExId{i}) = -vEx{i}(id, :)';
            % quadratic term in the objetive
                LP{i}.Model.Q = Q{i};
                LP{i}.Model.sense = 'minimize';
            % should always be feasible
                LP{i}.solve();
            % exchange rate
                R(:,i) = LP{i}.Solution.x(rxnExId{i});
            % flux vector
                fluxFBA{i} = LP{i}.Solution.x;
            % remove Q
                LP{i}.Model = rmfield(LP{i}.Model, 'Q');
            % update vEx
                id = id + 1;
                if id > size(vEx{i}, 1)
                    id = 1;
                end
                vEx{i}(id, :) = R(:,i)';
                t_v(id) = t;
            else
                fprintf('LP: %8.4f\tinfeasible\n', t);
            % infeasible, no activity
                R(:,i) = zeros(nnz(rxnExId{i}), 1);
                %R(:,i) = zeros(nnz(concat(rxnExId{i},xc_rxnExId)), 1);
                fluxFBA{i} = zeros(numel(LP{i}.Model.obj), 1);
            % can add cell lysis rate: rate constant = 3e-3 / min for
            % exponential-phase a E. coli K-12 mutant
            % Leduc, M. & van Heijenoort, J. (1980). Autolysis of Escherichia coli. Journal of bacteriology, 142(1), 52-59.
            end

        end

% constrain extracellular kinetics
        fluxKinetics_xc = arrayfun(@(x) x.eqs(C), kinetics_xc);
        fluxKinetics_xc(abs(fluxKinetics_xc) < feasTol) = 0;
        fluxKinetics_xc = fluxKinetics_xc(:);
        rxnChangeBounds = [kinetics_xc.rxn];
        LP_xc.Model.lb(rxnChangeBounds([kinetics_xc.type] == 'b' | [kinetics_xc.type] == 'l')) ...
            = fluxKinetics_xc([kinetics_xc.type] == 'b' | [kinetics_xc.type] == 'l');
        LP_xc.Model.ub(rxnChangeBounds([kinetics_xc.type] == 'b' | [kinetics_xc.type] == 'u')) ...
            = fluxKinetics_xc([kinetics_xc.type] == 'b' | [kinetics_xc.type] == 'u');
        %solve for extracellular reaction-------------------------------

        % setup general bounds

        % setup bounds by kinetics

        % solve model
        LP_xc.solve();

        % check solution
        if LP_xc.Solution.status == 1

            R_ex = LP_xc.Solution(xc_rxnExId);
            fluxFBA_xc = LP_xc.Solution.x;
        % extracellular kinetics
        else
            fprintf('LP: %8.4f\tinfeasible_extracellular\n', t);
            R_ex = zeros(nnz(xc_rxnExId), 1);
            fluxFBA_xc = zeros(numel(LP_xc.Model), 1);
        end
        % update extracellular concentrations
        % change = exchange by microbes + feed in - feed out
        % R: mmol/hr/gDW
        dCdt = R * C(xInEx) + (Cfeed - C) * dilutionRate + R_ex;
        % ***Sum above: (rate of each org * biomass of each) + same reactor factors
    end
end
