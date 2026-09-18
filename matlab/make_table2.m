function f = make_table2_updated()
% TABLE 2 ONLY.
% Updated manuscript robustness summary.
%
% Rows 1-11 reproduce the frozen saved robustness counts exactly.
% Row 12 adds the number of unanimously ordered strategy pairs among all
% 36 unordered total-burden strategy comparisons.
%
% No scenarios are rerun; this function only recounts saved outcomes.

d = mr.load_data();

sizes = [30 50 100];

headers = { ...
    'Criterion', ...
    'Reported (n=30)', ...
    'Reported (n=50)', ...
    'Reported (n=100)', ...
    'All infections (n=30)', ...
    'All infections (n=50)', ...
    'All infections (n=100)'};

nBaseCriteria = numel(d.criterion_labels);

if nBaseCriteria ~= 11
    error('Table2:CriterionCount', ...
        'Expected 11 saved robustness criteria, found %d.',nBaseCriteria);
end

rows = cell(nBaseCriteria+1,7);
rows(1:nBaseCriteria,1) = cellstr(d.criterion_labels);
rows{nBaseCriteria+1,1} = 'Unanimously ordered strategy pairs';

% -------------------------------------------------------------------------
% Existing 11 criteria
% -------------------------------------------------------------------------
for measure = 1:2

    for k = 1:3

        n = sizes(k);

        counts = mr.robustness(d,n,measure);

        if ~isequal(counts,d.expected_robustness(:,k,measure))
            error('Table2:FrozenSummaryMismatch', ...
                'Table 2 differs from the frozen saved robustness summary.');
        end

        for i = 1:nBaseCriteria

            rows{i,1+(measure-1)*3+k} = sprintf( ...
                '%d/%d (%.1f%%)', ...
                counts(i), ...
                n, ...
                counts(i)/n*100);
        end
    end
end

% -------------------------------------------------------------------------
% New row: unanimously ordered strategy pairs
%
% Denominator is 36 unordered strategy pairs, NOT the number of solutions.
% Ranking is based on strict member-wise comparison of total burden.
% -------------------------------------------------------------------------
for measure = 1:2

    for k = 1:3

        n = sizes(k);
        r = d.(sprintf('run%d',n));

        pairCount = count_unanimous_pairs_local(r.values,measure);

        rows{nBaseCriteria+1,1+(measure-1)*3+k} = sprintf( ...
            '%d/36 (%.1f%%)', ...
            pairCount, ...
            pairCount/36*100);
    end
end

% Frozen manuscript result: 6/36, 6/36, 4/36 for BOTH measures.
reportedPairCounts = [ ...
    count_unanimous_pairs_local(d.run30.values,1), ...
    count_unanimous_pairs_local(d.run50.values,1), ...
    count_unanimous_pairs_local(d.run100.values,1)];

allInfPairCounts = [ ...
    count_unanimous_pairs_local(d.run30.values,2), ...
    count_unanimous_pairs_local(d.run50.values,2), ...
    count_unanimous_pairs_local(d.run100.values,2)];

if ~isequal(reportedPairCounts,[6 6 4])
    error('Table2:ReportedPairCountMismatch', ...
        'Expected reported unanimous-pair counts [6 6 4], found [%d %d %d].', ...
        reportedPairCounts(1),reportedPairCounts(2),reportedPairCounts(3));
end

if ~isequal(allInfPairCounts,[6 6 4])
    error('Table2:AllInfectionsPairCountMismatch', ...
        'Expected all-infections unanimous-pair counts [6 6 4], found [%d %d %d].', ...
        allInfPairCounts(1),allInfPairCounts(2),allInfPairCounts(3));
end

% -------------------------------------------------------------------------
% Render
% -------------------------------------------------------------------------
f = mr.tablefigure( ...
    'Table2', ...
    headers, ...
    rows, ...
    [.30 repmat(.116,1,6)], ...
    6.4, ...
    7.6);

f.UserData.ManuscriptTable = rows;
f.UserData.ReportedUnanimousPairCounts = reportedPairCounts;
f.UserData.AllInfectionsUnanimousPairCounts = allInfPairCounts;

mr.save(f,'Table2','tables');

end


function nPairs = count_unanimous_pairs_local(values,measure)
% Count unanimously ordered unordered strategy pairs for total burden.

E = squeeze(values(:,:,measure,5));

nStrategies = size(E,2);

nPairs = 0;

for i = 1:(nStrategies-1)

    for j = (i+1):nStrategies

        p = mean(E(:,i) > E(:,j));

        if p <= 1e-12 || p >= 1-1e-12
            nPairs = nPairs + 1;
        end
    end
end

end
