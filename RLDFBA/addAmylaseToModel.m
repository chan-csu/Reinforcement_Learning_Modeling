function [model, xc_model, metAbbrs, rxnAbbrs] = addAmylaseToModel(model, g_or_mmol, starchChainLength, aaSeq, rxnAbbrs, metAbbrs, compAbbrs)

% % protein sequence for beta-amylase (https://www.uniprot.org/uniprot/P36924)
% aaSeq = ['MKNQFQYCCIVILSVVMLFVSLLIPQASSAAVNGKGMNPDYKAYLMAPLKKIPEVTNWETFEND' ...
%     'LRWAKQNGFYAITVDFWWGDMEKNGDQQFDFSYAQRFAQSVKNAGMKMIPIISTHQCGGNVGDDCNVPI' ...
%     'PSWVWNQKSDDSLYFKSETGTVNKETLNPLASDVIRKEYGELYTAFAAAMKPYKDVIAKIYLSGGPAGE' ...
%     'LRYPSYTTSDGTGYPSRGKFQAYTEFAKSKFRLWVLNKYGSLNEVNKAWGTKLISELAILPPSDGEQFL' ...
%     'MNGYLSMYGKDYLEWYQGILENHTKLIGELAHNAFDTTFQVPIGAKIAGVHWQYNNPTIPHGAEKPAGY' ...
%     'NDYSHLLDAFKSAKLDVTFTCLEMTDKGSYPEYSMPKTLVQNIATLANEKGIVLNGENALSIGNEEEYK' ...
%     'RVAEMAFNYNFAGFTLLRYQDVMYNNSLMGKFKDLLGVTPVMQTIVVKNVPTTIGDTVYITGNRAELGS' ...
%     'WDTKQYPIQLYYDSHSNDWRGNVVLPAERNIEFKAFIKSKDGTVKSWQTIQQSWNPVPLKTTSHTSSW'];



[abbrsDefault, abbrs] = deal(struct());
abbrsDefault.rxns = {'amylase', 'AMYLASE'; ...
    'amylaseTransport', 'AMYLt'; ...
    'amylaseProduction', 'AMYLp'; ...
    'amylaseExchange', 'EX_amylase(e)'; ...
    'amylaseDegradation', 'AMYLd'; ...
    'amylase_dExchange', 'EX_amylase_d(e)'; ...
    'starchExchange', 'EX_starch(e)'; ...
    'glucoseExchange', 'EX_glc__D(e)'}; ...
    %'glucoseExchange', 'EX_glc__D(u)'};
abbrsDefault.mets = {'glucose', 'glc__D'; ...
    'starch', 'starch'; ...
    'h2o', 'h2o';
    'amylase', 'amylase'; ...
    'amylase_d', 'amylase_d'};  % degraded amylase (amino acids, inactive, whatever)
abbrsDefault.comps = {'c', '[c]'; ...  % cytoplasm
    'e', '[e]'; ...  % extracellular space of the organism
    'u', '[u]'};  % extracellular space of the medium, only useful for glucose, to distinguish organism uptake and release from starch degradation

if nargin < 8 || isempty(compAbbrs)
    compAbbrs = repmat({''}, 0, 2);
elseif isstruct(compAbbrs)
    compAbbrs = [fieldnames(compAbbrs), struct2cell(compAbbrs)];
end
if nargin < 7 || isempty(metAbbrs)
    metAbbrs = repmat({''}, 0, 2);
elseif isstruct(metAbbrs)
    metAbbrs = [fieldnames(metAbbrs), struct2cell(metAbbrs)];
end
if nargin < 6 || isempty(rxnAbbrs)
    rxnAbbrs = repmat({''}, 0, 2);
elseif isstruct(rxnAbbrs)
    rxnAbbrs = [fieldnames(rxnAbbrs), struct2cell(rxnAbbrs)];
end
if nargin < 5 || isempty(aaSeq)
    % uniprot: B8Y1H0 (Bacillus subtilis alpha-amylase)
    aaSeq = ['MFAKRFKTSLLPLFAGFLLLFHLVLAGPAAASAETANKSNELTAPSIKSGTILHAWNWSF' ...
        'NTLKHNMKDIHDAGYTAIQTSPINQVKEGNQGDKSMSNWYWLYHPTSYQIGNRYLGTEQE' ...
        'FKEMCAAAEEYGIKVIVDAVINHTTSDYAAISNEVKSIPNWTHGNTQIKNWSDRWDVTQN' ...
        'SLLGLYDWNTQNTQVQSYLKRFLERALNDGADGFRFDAAKHIELPDDGSYGSQFWPNITN' ...
        'TSAEFQYGEILQDSASRDAAYANYMDVTASNYGHSIRSALKNRNLGVSNISHYASDVSAD' ...
        'KLVTWVESHDTYANDDEESTWMSDDDIRLGWAVIASRSGSTPLFFSRPEGGGNGVRFPGK' ...
        'SQIGDRGSALFEDQAITAVNRFHNVMAGQHEELSNPNGNNQIFMNQRGSHGVVLANAGSS' ...
        'SVSINTATKLPDGRYDNKAGAGSFQVNDGKLTGTINARSVAVLYPDDIEIRCNTFFQ'];
end
if nargin < 4 || isempty(starchChainLength)
    % Hydrolysis catalyzed by amylase. Assume an average chain length of 20
    % (Wang, Ya-Jane & Wang, Linfeng. (2003). Physiochemical properties of common and waxy corn starches oxidized by different levels of sodium hypochlorite. Carbohydrate Polymers. 52. 207-217. 10.1016/S0144-8617(02)003041.)
    starchChainLength = 20;
end
if nargin < 3 || isempty(g_or_mmol)
    % use mmol or gram as the unit for the enzyme amylase
    g_or_mmol = 'g';
end

[abbrs.rxns, abbrs.mets, abbrs.comps] = deal(rxnAbbrs, metAbbrs, compAbbrs);
for str = {'rxns', 'mets', 'comps'}
    yn = ismember(abbrsDefault.(str{1})(:, 1), abbrs.(str{1})(:, 1));
    abbrs.(str{1}) = [abbrs.(str{1}); abbrsDefault.(str{1})(~yn, :)];
    abbrs.(str{1}) = abbrs.(str{1})';
    abbrs.(str{1}) = struct(abbrs.(str{1}){:});
end
% amylase reaction: starch[e] + (n - 1) h2o[e] -> n glc__D[e]
amylaseRxn = sprintf('%1$s%4$s + %6$d %2$s%4$s -> %7$d %3$s%5$s', ...
    abbrs.mets.starch, abbrs.mets.h2o, abbrs.mets.glucose, abbrs.comps.e, abbrs.comps.e, starchChainLength - 1, starchChainLength);

% create the protein production reaction for amylase
% ('help aaSeqtoRxn' to look at how it works)
abbrs.mets.protein = abbrs.mets.amylase;
[rxnFormulaInMole, rxnFormulaInGram, proteinMW] = aaSeqToRxn(aaSeq, [], abbrs.mets, abbrs.comps);
if strcmp(g_or_mmol, 'mmol')
    rxnFormula = rxnFormulaInMole;
else
    rxnFormula = rxnFormulaInGram;
end
fprintf('Reaction formula (protein in %s):\n%s\n', g_or_mmol, rxnFormula)
% notice that usual proteins have large molecular weight, so 1 mmol of a
% protein can have 10 ~ 100 g of mass.
fprintf('Protein molar mass: %.4f g/mmol\n', proteinMW / 1000)

%% Add metabolites
xc_model = createModel();
xc_model = addMetabolite(xc_model, [abbrs.mets.starch abbrs.comps.e], 'metName', 'starch');
xc_model = addMetabolite(xc_model, [abbrs.mets.amylase abbrs.comps.e], 'metName', 'amylase');
xc_model = addMetabolite(xc_model, [abbrs.mets.amylase_d abbrs.comps.e], 'metName', 'degraded amylase');
xc_model = addMetabolite(xc_model, [abbrs.mets.glucose abbrs.comps.e], 'metName', 'glucose in the medium');
xc_model = addMetabolite(xc_model, [abbrs.mets.h2o abbrs.comps.e], 'metName', 'water in the medium');
%model = addMetabolite(model, [abbrs.mets.starch abbrs.comps.e], 'metName', 'starch');
model = addMetabolite(model, [abbrs.mets.amylase abbrs.comps.e], 'metName', 'amylase');
%model = addMetabolite(model, [abbrs.mets.amylase_d abbrs.comps.e], 'metName', 'degraded amylase');
%model = addMetabolite(model, [abbrs.mets.glucose abbrs.comps.u], 'metName', 'glucose in the medium');

%% Add reactions
% add the protein production reaction (where the protein is in mole or gram)
% ('help addReaction' to look at how it works)
model = addReaction(model, abbrs.rxns.amylaseProduction, 'reactionFormula', rxnFormula, ...
    'reactionName', sprintf('Amylase production (in %s)', g_or_mmol));

%%% The two reactions below are added to make sure the protein production can
%%% happen. If not added, 'amylase[c]' is a dead end and no reaction consumes
%%% it so the production flux must be 0.
% add transport reaction for amylase to go to the extracellular space.
model = addReaction(model, abbrs.rxns.amylaseTransport, 'reactionFormula', ...
    sprintf('%1$s%2$s -> %1$s%3$s',abbrs.mets.amylase, abbrs.comps.c, abbrs.comps.e), ...
    'reactionName', 'Amylase transport');
% add amylase degradation reaction
xc_model = addReaction(xc_model, abbrs.rxns.amylaseDegradation, 'reactionFormula', ...
    [abbrs.mets.amylase, abbrs.comps.e ' -> ' abbrs.mets.amylase_d, abbrs.comps.e], ...
    'reactionName', 'Amylase degradation');
% add exchange reaction for amylase to get out of the system
model = addReaction(model, abbrs.rxns.amylaseExchange, 'reactionFormula', [abbrs.mets.amylase, abbrs.comps.e ' ->'], ...
    'reactionName', 'Exchange reaction for amylase');
xc_model = addReaction(xc_model, abbrs.rxns.amylaseExchange, 'reactionFormula', [abbrs.mets.amylase, abbrs.comps.e ' ->'], ...
    'reactionName', 'Exchange reaction for amylase');
xc_model = addReaction(xc_model, abbrs.rxns.amylase_dExchange, 'reactionFormula', [abbrs.mets.amylase_d, abbrs.comps.e ' ->'], ...
    'reactionName', 'Exchange reaction for degraded amylase');
% add amylase reaction that breaks down starch
xc_model = addReaction(xc_model, abbrs.rxns.amylase, 'reactionFormula', amylaseRxn, ...
    'reactionName', 'Amylase');
%%model = addReaction(model, abbrs.rxns.amylase, 'reactionFormula', amylaseRxn, ...
    %%'reactionName', 'Amylase');
xc_model = addReaction(xc_model, abbrs.rxns.starchExchange, 'reactionFormula', [abbrs.mets.starch, abbrs.comps.e ' <=>'], ...
    'reactionName', 'Exchange reaction for starch');
% change the original exchange reaction for glucose into exchange between e and u
% r = findRxnIDs(xc_model, abbrs.rxns.glucoseOrgExchange);
% m = findMetIDs(xc_model, [abbrs.mets.glucose abbrs.comps.u]);
% xc_model.S(m, r) = 1;
% repeated for intracellular model
% r = findRxnIDs(model, abbrs.rxns.glucoseOrgExchange);
% m = findMetIDs(model, [abbrs.mets.glucose abbrs.comps.u]);
% model.S(m, r) = 1;
% add exchange reaction for glucose in u
xc_model = addReaction(xc_model, abbrs.rxns.glucoseExchange, 'reactionFormula', ...
   [abbrs.mets.glucose abbrs.comps.e ' <=>'], 'reactionName', 'Exchange reaction for glucose in the medium');

xc_model = addReaction(xc_model, 'EX_h2o(e)', 'reactionFormula', ...
   'h2o[e] <=>', 'reactionName', 'Exchange reaction for H2O in the medium');


[metAbbrs, rxnAbbrs] = deal(struct());
metAbbrs.starch = [abbrs.mets.starch abbrs.comps.e];
metAbbrs.glucose_e = [abbrs.mets.glucose abbrs.comps.e];
metAbbrs.glucose_u = [abbrs.mets.glucose abbrs.comps.u];
metAbbrs.amylase = [abbrs.mets.amylase abbrs.comps.e];
metAbbrs.amylase_d = [abbrs.mets.amylase_d abbrs.comps.e];
metAbbrs.h2o = [abbrs.mets.h2o abbrs.comps.e];
rxnAbbrs = abbrs.rxns;
