function [rxnFormulaInMole, rxnFormulaInGram, proteinMW] = aaSeqToRxn(aaSeq, atpCostPerBond, metAbbrs, compAbbrs)
% Given a protein sequence, the ATP cost for polymerization per peptide
% bond, return the reaction formula representing the overall production
% of the protein. The IDs used in the reaction are BiGG IDs.
%
% USAGE:
%    [rxnFormulaInMole, rxnFormulaInGram, proteinMW] = aaSeqToRxn(aaSeq, atpCostPerBond, proteinMetID)
%
% INPUT:
%    aaSeq                   protein sequence (sequence only, i.e., no header)
%
% OPTIONAL INPUTS:
%    atpCostPerBond          the ATP cost for polymerization per peptide bond (default 4.5)
%    proteinMetID            the met ID for the protein (default 'protein[c]')
%
% OUTPUTS:
%    rxnFormulaInMole        protein production formula. The protein is in
%                            mmol (assuming other mets are in mmol)
%    rxnFormulaInGram        protein production formula. The protein is in
%                            gram (assuming other mets are in mmol, the usual
%                            representation in genome-scale metabolic models)
%    proteinMW               molar mass for the protein

% default metabolite abbreviations for constructing the formula (BiGG IDs)
metAbbrsDefault = {'ala', 'ala__L'; ...
    'arg', 'arg__L'; ...
    'asn', 'asn__L'; ...
    'asp', 'asp__L'; ...
    'cys', 'cys__L'; ...
    'glu', 'glu__L'; ...
    'gln', 'gln__L'; ...
    'gly', 'gly'; ...
    'his', 'his__L'; ...
    'ile', 'ile__L'; ...
    'leu', 'leu__L'; ...
    'lys', 'lys__L'; ...
    'met', 'met__L'; ...
    'phe', 'phe__L'; ...
    'pro', 'pro__L'; ...
    'ser', 'ser__L'; ...
    'thr', 'thr__L'; ...
    'trp', 'trp__L'; ...
    'tyr', 'tyr__L'; ...
    'val', 'val__L'; ...
    'h2o', 'h2o'; ...
    'atp', 'atp'; ...
    'adp', 'adp'; ...
    'pi', 'pi'; ...
    'h', 'h'; ...
    'protein', 'protein'};
% default compartment abbreviation appended to the metabolite abbreviations
compAbbrsDefault = {'c', '[c]'};

if nargin < 2 || isempty(atpCostPerBond)
    atpCostPerBond = 4.5;
end
if nargin < 3 || isempty(metAbbrs)
    metAbbrs = repmat({''}, 0, 2);
elseif isstruct(metAbbrs)
    metAbbrs = [fieldnames(metAbbrs), struct2cell(metAbbrs)];
end
if nargin < 4 || isempty(compAbbrs)
    compAbbrs = repmat({''}, 0, 2);
elseif isstruct(compAbbrs)
    compAbbrs = [fieldnames(compAbbrs), struct2cell(compAbbrs)];
end
% add default met and comp abbreviations if not in the input structure/cell
yn = ismember(metAbbrsDefault(:, 1), metAbbrs(:, 1));
metAbbrs = [metAbbrs; metAbbrsDefault(~yn, :)];
metAbbrs = metAbbrs';
metAbbrs= struct(metAbbrs{:});  % convert to structure
yn = ismember(compAbbrsDefault(:, 1), compAbbrs(:, 1));
compAbbrs = [compAbbrs; compAbbrsDefault(~yn, :)];
compAbbrs = compAbbrs';
compAbbrs= struct(compAbbrs{:});

% BiGG IDs for proteogenic amino acids
aaID = cellfun(@(x) [metAbbrs.(x), compAbbrs.c], metAbbrsDefault(1:20, 1), 'UniformOutput', false);
% the corresponding sequence code
aaCode = 'ARNDCEQGHILKMFPSTWYV';
% the corresponding molar mass
aaMW = [89.09318; 175.2089; 132.11792; 132.09474; 121.15818; 146.12132; ...
    146.1445; 75.06660; 155.15456; 131.17292; 131.17292; 147.1955; ...
    149.21134; 165.18914; 115.13046; 105.09258; 119.11916; 204.22518; 181.18854;117.14634];

% obtain sequence counts
aaCount = zeros(numel(aaCode), 1);
for j = 1:numel(aaCode)
    aaCount(j) = sum(aaSeq == aaCode(j));
end

% molecular weight for the protein: sum of the weights for all amino acids
% minus the length - 1 copies of water being released
proteinMW = aaCount' * aaMW - (length(aaSeq) - 1) * 18.0153;

% total ATP cost per mmol of protein
atpCost = atpCostPerBond * (length(aaSeq) - 1);
% string format
atpCostStr = num2str(atpCost, 16);
% total ATP cost per gram of protein
atpCostStrGram = num2str(atpCost / (proteinMW / 1000), 16);
% water required per mmol of protein
water = (atpCostPerBond - 1) * (length(aaSeq) - 1);
% formula for producing 1 mmol of protein
rxnFormulaInMole = [atpCostStr ' ' metAbbrs.atp compAbbrs.c ' + ' num2str(water) ' ' metAbbrs.h2o compAbbrs.c];
% formula for producing 1 g of protein
rxnFormulaInGram = [atpCostStrGram ' ' metAbbrs.atp compAbbrs.c ' + '  num2str(water / (proteinMW / 1000), 16) ' ' metAbbrs.h2o compAbbrs.c];
for j = 1:numel(aaCode)
    rxnFormulaInMole = [rxnFormulaInMole ' + ' num2str(aaCount(j)) ' ' aaID{j}];
    rxnFormulaInGram = [rxnFormulaInGram ' + ' num2str(aaCount(j) / (proteinMW / 1000), 16) ' ' aaID{j}];
end
rxnFormulaInMole = [rxnFormulaInMole ' -> ' metAbbrs.protein compAbbrs.c ' + ' ...
    atpCostStr ' ' metAbbrs.adp compAbbrs.c ' + ' ...
    atpCostStr ' ' metAbbrs.pi compAbbrs.c ' + '  ...
    atpCostStr ' ' metAbbrs.h compAbbrs.c];
rxnFormulaInGram = [rxnFormulaInGram ' -> ' metAbbrs.protein compAbbrs.c ' + ' ...
    atpCostStrGram ' ' metAbbrs.adp compAbbrs.c ' + ' ....
    atpCostStrGram ' ' metAbbrs.pi compAbbrs.c ' + '  ....
    atpCostStrGram ' ' metAbbrs.h compAbbrs.c];

end
