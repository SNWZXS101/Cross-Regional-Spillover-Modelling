%% plot_fig4_atlas_panels.m
% Directly runnable MATLAB R2024b script.
%
% Generates the four panels of the atlas-version Fig. 4 as FOUR SEPARATE
% editable MATLAB figures:
%
%   Fig4_a.fig / Fig4_a.png   3x3 regional effect phase-portrait atlas
%   Fig4_b.fig / Fig4_b.png   outcome-class fingerprint
%   Fig4_c.fig / Fig4_c.png   HH-LL distribution summary
%   Fig4_d.fig / Fig4_d.png   integrated median-effect matrix
%
% Output folder:
%   UDEcodes\MainResults\matlab\figures
%
% Requires the existing project utilities:
%   mr.load_data()
%   mr.style()
%   mr.symlog()
%   mr.signedaxis()

clearvars;
close all;
clc;

%% ------------------------------------------------------------------------
% Load data
% -------------------------------------------------------------------------
d = mr.load_data();
s = mr.style();
v = d.run100.values;

Nsol = size(v,1);

% Robust strategy-name conversion
if isstring(d.strategies)
    strategy_codes = cellstr(d.strategies(:));
elseif iscell(d.strategies)
    strategy_codes = d.strategies(:);
else
    strategy_codes = cellstr(string(d.strategies(:)));
end

% Strategy order:
% 1 NN, 2 HN, 3 LN, 4 NH, 5 NL, 6 HH, 7 HL, 8 LH, 9 LL
orderMat = [1 4 5; ...
            2 6 7; ...
            3 8 9];

% Reported-case effects, in thousands
GD  = squeeze(v(:,:,1,3))/1000;
GX  = squeeze(v(:,:,1,4))/1000;
TOT = squeeze(v(:,:,1,5))/1000;

medGD  = squeeze(median(v(:,:,1,3),1))/1000;
medGX  = squeeze(median(v(:,:,1,4),1))/1000;
medTOT = squeeze(median(v(:,:,1,5),1))/1000;

Mgd  = medGD(orderMat);
Mgx  = medGX(orderMat);
Mtot = medTOT(orderMat);

% HH - LL paired differences
G_reported = (v(:,6,1,5) - v(:,9,1,5))/1000;
G_allinf   = (v(:,6,2,5) - v(:,9,2,5))/1000;

%% ------------------------------------------------------------------------
% Shared style
% -------------------------------------------------------------------------
INK     = [0.08 0.13 0.17];
SUBINK  = [0.35 0.39 0.42];
WHITE   = [1 1 1];
GRIDC   = [0.77 0.79 0.80];
POINTC  = [0.20 0.34 0.46];

benefit = s.benefit;
harm    = s.harm;
neutral = s.neutral;

class_names = { ...
    'win-win', ...
    'self-harm / neighbor-benefit', ...
    'burden shifting', ...
    'lose-lose', ...
    'neutral', ...
    'mixed with neutral'};

class_short = { ...
    'WW', ...
    'SH/NB', ...
    'BS', ...
    'LL', ...
    'NEU', ...
    'MIX'};

class_labels = { ...
    'Win-win', ...
    'Self-harm / neighbour benefit', ...
    'Burden shifting', ...
    'Lose-lose', ...
    'Neutral', ...
    'Mixed with neutral'};

class_colors = [ ...
    benefit; ...
    0.56 0.72 0.77; ...
    0.86 0.75 0.52; ...
    harm; ...
    neutral; ...
    0.80 0.84 0.84];

quad_fill = class_colors(1:4,:);
quad_fill = min(1,0.60*quad_fill + 0.40);

%% ------------------------------------------------------------------------
% Derived objects shared by multiple panels
% -------------------------------------------------------------------------
% Outcome fingerprint
strategy_labels_nonbase = strategy_codes(2:end);
nStrategies = numel(strategy_labels_nonbase);

overallD = compute_baseline_distance_local(d);
[~,solOrder] = sort(overallD,'ascend');

class_idx = nan(nStrategies,Nsol);

for i = 1:nStrategies
    rawClass = string(d.run100.classification(:,i+1,1));
    rawClass = rawClass(solOrder);

    for k = 1:numel(class_names)
        class_idx(i,rawClass == string(class_names{k})) = k;
    end
end

class_idx(~isfinite(class_idx)) = numel(class_names);

% Common atlas limits
allXY = [GD(:); GX(:)];
allXY = allXY(isfinite(allXY));

maxAbs = max(abs(allXY));
maxAbs = max(maxAbs,2);

limRaw = [-maxAbs maxAbs];

thrAtlas = 0.1;
limAtlas = mr.symlog(limRaw,thrAtlas);

% Integrated matrix colour range
limitD = max(abs([Mgd(:); Mgx(:); Mtot(:)]));
zLimD  = mr.symlog([-limitD limitD],1,1);

%% ------------------------------------------------------------------------
% Output folder
% -------------------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));

if isempty(script_dir)
    script_dir = pwd;
end

output_dir = fullfile(script_dir,'Fig4');

if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

%% =========================================================================
% PANEL a — 3x3 regional effect phase-portrait atlas
% =========================================================================
figA = figure( ...
    'Color','w', ...
    'Units','pixels', ...
    'Position',[80 40 1320 1180], ...
    'Name','Fig4_a', ...
    'NumberTitle','off');

set(figA,'Renderer','painters');

annotation(figA,'textbox',[0.020 0.950 0.035 0.035], ...
    'String','a', ...
    'EdgeColor','none', ...
    'FontName','Arial', ...
    'FontSize',30, ...
    'FontWeight','bold', ...
    'Color','k');

annotation(figA,'textbox',[0.19 0.955 0.63 0.030], ...
    'String','Regional effect phase portraits across 100 admissible solutions', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',20, ...
    'FontWeight','bold', ...
    'Color',INK);

annotation(figA,'textbox',[0.10 0.915 0.80 0.030], ...
    'String',['x = GD avoided cases; y = GX avoided cases. ' ...
              'Quadrants indicate the signs of regional effects; point colours show strategy-aware outcome classes.'], ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',13.5, ...
    'Color',SUBINK);

% Grid geometry
leftA   = 0.125;
bottomA = 0.135;
widthA  = 0.745;
heightA = 0.700;

gapxA = 0.025;
gapyA = 0.035;

smallW = (widthA - 2*gapxA)/3;
smallH = (heightA - 2*gapyA)/3;

colLab = {'GX = N','GX = H','GX = L'};
rowLab = {'GD = N','GD = H','GD = L'};

for rr = 1:3
    for cc = 1:3

        sid = orderMat(rr,cc);

        px = leftA + (cc-1)*(smallW+gapxA);
        py = bottomA + (3-rr)*(smallH+gapyA);

        ax = axes(figA,'Position',[px py smallW smallH]);
        hold(ax,'on');

        % quadrant backgrounds
        patch(ax,[0 limAtlas(2) limAtlas(2) 0], ...
            [0 0 limAtlas(2) limAtlas(2)], ...
            quad_fill(1,:), ...
            'FaceAlpha',0.16, ...
            'EdgeColor','none');

        patch(ax,[limAtlas(1) 0 0 limAtlas(1)], ...
            [0 0 limAtlas(2) limAtlas(2)], ...
            quad_fill(2,:), ...
            'FaceAlpha',0.16, ...
            'EdgeColor','none');

        patch(ax,[0 limAtlas(2) limAtlas(2) 0], ...
            [limAtlas(1) limAtlas(1) 0 0], ...
            quad_fill(3,:), ...
            'FaceAlpha',0.16, ...
            'EdgeColor','none');

        patch(ax,[limAtlas(1) 0 0 limAtlas(1)], ...
            [limAtlas(1) limAtlas(1) 0 0], ...
            quad_fill(4,:), ...
            'FaceAlpha',0.16, ...
            'EdgeColor','none');

        xraw = double(GD(:,sid));
        yraw = double(GX(:,sid));

        xT = mr.symlog(xraw,thrAtlas);
        yT = mr.symlog(yraw,thrAtlas);

        % Use the stored, strategy-aware outcome classification.  A given
        % sign quadrant does not have the same policy meaning for unilateral
        % and bilateral strategies, so classifying points from x/y signs alone
        % would mislabel NH/NL and some bilateral outcomes.
        rawClassPoints = string(d.run100.classification(:,sid,1));
        point_cls = numel(class_names)*ones(size(rawClassPoints));
        for kk = 1:numel(class_names)
            point_cls(rawClassPoints == string(class_names{kk})) = kk;
        end

        for kk = 1:numel(class_names)
            ok = (point_cls == kk);

            if any(ok)
                h = scatter(ax,xT(ok),yT(ok),32, ...
                    'MarkerFaceColor',class_colors(kk,:), ...
                    'MarkerEdgeColor',darken_local(class_colors(kk,:),0.28), ...
                    'LineWidth',0.40);

                try
                    h.MarkerFaceAlpha = 0.74;
                    h.MarkerEdgeAlpha = 0.38;
                catch
                end
            end
        end

        % origin lines
        plot(ax,[0 0],limAtlas,'-','Color',GRIDC,'LineWidth',0.95);
        plot(ax,limAtlas,[0 0],'-','Color',GRIDC,'LineWidth',0.95);

        % median direction and median point
        xmed = median(xraw,'omitnan');
        ymed = median(yraw,'omitnan');

        xmedT = mr.symlog(xmed,thrAtlas);
        ymedT = mr.symlog(ymed,thrAtlas);

        plot(ax,[0 xmedT],[0 ymedT],'-', ...
            'Color',POINTC, ...
            'LineWidth',1.65);

        plot(ax,xmedT,ymedT,'o', ...
            'MarkerFaceColor','w', ...
            'MarkerEdgeColor',POINTC, ...
            'MarkerSize',7.2, ...
            'LineWidth',1.45);

        % dominant outcome class from stored classification
        rawClassFull = string(d.run100.classification(:,sid,1));

        counts = zeros(1,numel(class_names));

        for kk = 1:numel(class_names)
            counts(kk) = sum(rawClassFull == string(class_names{kk}));
        end

        [mxCount,mxIdx] = max(counts);

        text(ax,0.03,0.96, ...
            sprintf('%d%% %s',round(100*mxCount/Nsol),class_short{mxIdx}), ...
            'Units','normalized', ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','top', ...
            'FontName','Arial', ...
            'FontSize',12.0, ...
            'FontWeight','bold', ...
            'Color',INK);

        text(ax,0.97,0.05, ...
            sprintf('med total %+.1f',medTOT(sid)), ...
            'Units','normalized', ...
            'HorizontalAlignment','right', ...
            'VerticalAlignment','bottom', ...
            'FontName','Arial', ...
            'FontSize',11.5, ...
            'Color',SUBINK);

        text(ax,0.97,0.93,strategy_codes{sid}, ...
            'Units','normalized', ...
            'HorizontalAlignment','right', ...
            'VerticalAlignment','top', ...
            'FontName','Arial', ...
            'FontSize',14.0, ...
            'FontWeight','bold', ...
            'Color',POINTC);

        set(ax, ...
            'XLim',limAtlas, ...
            'YLim',limAtlas, ...
            'FontName','Arial', ...
            'FontSize',11.0, ...
            'LineWidth',0.95, ...
            'TickDir','out', ...
            'Box','off', ...
            'XColor',INK, ...
            'YColor',INK);

        mr.signedaxis(ax,'x',limRaw,thrAtlas);
        mr.signedaxis(ax,'y',limRaw,thrAtlas);

        if rr < 3
            set(ax,'XTickLabel',[]);
        end

        if cc > 1
            set(ax,'YTickLabel',[]);
        end

        if rr == 1
            annotation(figA,'textbox', ...
                [px py+smallH+0.006 smallW 0.025], ...
                'String',colLab{cc}, ...
                'EdgeColor','none', ...
                'HorizontalAlignment','center', ...
                'FontName','Arial', ...
                'FontSize',14.0, ...
                'FontWeight','bold', ...
                'Color',INK);
        end

        if cc == 1
            annotation(figA,'textbox', ...
                [px-0.074 py+0.5*smallH-0.012 0.065 0.025], ...
                'String',rowLab{rr}, ...
                'EdgeColor','none', ...
                'HorizontalAlignment','right', ...
                'FontName','Arial', ...
                'FontSize',14.0, ...
                'FontWeight','bold', ...
                'Color',INK);
        end
    end
end

% shared axis labels
annotation(figA,'textbox',[leftA+0.16 bottomA-0.073 widthA-0.32 0.028], ...
    'String','GD avoided cases (10^3, symlog scale)', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',16.0, ...
    'Color',INK);

annotation(figA,'textbox',[leftA-0.095 bottomA+0.13 0.028 heightA-0.26], ...
    'String','GX avoided cases (10^3, symlog scale)', ...
    'Rotation',90, ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',16.0, ...
    'Color',INK);

% compact legend
legAxA = axes(figA,'Position',[0.250 0.045 0.540 0.045]);
hold(legAxA,'on');
axis(legAxA,'off');

for kk = 1:4
    plot(legAxA,kk,1,'s', ...
        'MarkerSize',9, ...
        'MarkerFaceColor',class_colors(kk,:), ...
        'MarkerEdgeColor','none');
end

text(legAxA,1.18,1,'Win-win', ...
    'FontName','Arial','FontSize',11.5,'VerticalAlignment','middle','Color',INK);
text(legAxA,2.18,1,'Self-harm / neighbour benefit', ...
    'FontName','Arial','FontSize',11.5,'VerticalAlignment','middle','Color',INK);
text(legAxA,3.18,1,'Burden shifting', ...
    'FontName','Arial','FontSize',11.5,'VerticalAlignment','middle','Color',INK);
text(legAxA,4.18,1,'Lose-lose', ...
    'FontName','Arial','FontSize',11.5,'VerticalAlignment','middle','Color',INK);

xlim(legAxA,[0.5 5.2]);
ylim(legAxA,[0.7 1.3]);

save_panel_local(figA,output_dir,'Fig4_a');

%% =========================================================================
% PANEL b — outcome-class fingerprint
% =========================================================================
figB = figure( ...
    'Color','w', ...
    'Units','pixels', ...
    'Position',[100 80 1500 760], ...
    'Name','Fig4_b', ...
    'NumberTitle','off');

set(figB,'Renderer','painters');

annotation(figB,'textbox',[0.018 0.935 0.035 0.040], ...
    'String','b', ...
    'EdgeColor','none', ...
    'FontName','Arial', ...
    'FontSize',30, ...
    'FontWeight','bold', ...
    'Color','k');

axB = axes(figB,'Position',[0.085 0.175 0.735 0.660]);

imagesc(axB,1:Nsol,1:nStrategies,class_idx);

colormap(axB,class_colors);
clim(axB,[0.5 numel(class_names)+0.5]);

set(axB, ...
    'YDir','reverse', ...
    'FontName','Arial', ...
    'FontSize',17, ...
    'LineWidth',1.0, ...
    'TickDir','out', ...
    'Box','off', ...
    'XColor',INK, ...
    'YColor',INK);

yticks(axB,1:nStrategies);
yticklabels(axB,strategy_labels_nonbase);

xlim(axB,[0.5 Nsol+23]);
xticks(axB,[1 25 50 75 100]);

xlabel(axB,'Admissible solutions ordered by functional distance from baseline', ...
    'FontSize',19, ...
    'Color',INK);

ylabel(axB,'Strategy', ...
    'FontSize',19, ...
    'Color',INK);

hold(axB,'on');

for y = 1.5:1:(nStrategies-0.5)
    yline(axB,y,'Color','w','LineWidth',0.8);
end

for x = [25.5 50.5 75.5]
    xline(axB,x,':','Color',[0.78 0.78 0.78],'LineWidth',0.8);
end

for i = 1:nStrategies

    rowvals = class_idx(i,:);
    counts = zeros(1,numel(class_names));

    for kk = 1:numel(class_names)
        counts(kk) = sum(rowvals == kk);
    end

    [mxCount,mxIdx] = max(counts);

    text(axB,Nsol+3,i, ...
        sprintf('%d%% %s',round(100*mxCount/Nsol),lower(class_labels{mxIdx})), ...
        'HorizontalAlignment','left', ...
        'VerticalAlignment','middle', ...
        'FontName','Arial', ...
        'FontSize',14.0, ...
        'Color',INK);
end

present = false(1,numel(class_names));

for kk = 1:numel(class_names)
    present(kk) = any(class_idx(:)==kk);
end

hLegB = gobjects(0);
labLegB = {};

for kk = find(present)

    hLegB(end+1) = plot(axB,nan,nan,'s', ...
        'MarkerSize',10, ...
        'MarkerFaceColor',class_colors(kk,:), ...
        'MarkerEdgeColor','none'); %#ok<AGROW>

    labLegB{end+1} = class_labels{kk}; %#ok<AGROW>
end

lgB = legend(axB,hLegB,labLegB, ...
    'NumColumns',2, ...
    'Box','off', ...
    'FontName','Arial', ...
    'FontSize',13.0);

lgB.Position = [0.545 0.855 0.330 0.075];

save_panel_local(figB,output_dir,'Fig4_b');

%% =========================================================================
% PANEL c — HH minus LL distribution summary
% =========================================================================
figC = figure( ...
    'Color','w', ...
    'Units','pixels', ...
    'Position',[120 100 1450 730], ...
    'Name','Fig4_c', ...
    'NumberTitle','off');

set(figC,'Renderer','painters');

annotation(figC,'textbox',[0.018 0.935 0.035 0.040], ...
    'String','c', ...
    'EdgeColor','none', ...
    'FontName','Arial', ...
    'FontSize',30, ...
    'FontWeight','bold', ...
    'Color','k');

axC = axes(figC,'Position',[0.120 0.160 0.815 0.690]);
hold(axC,'on');

Gc = {G_reported(:),G_allinf(:)};
rowY = [2 1];

yTickC = [1 2];
yTickLabelsC = {'All infections','Reported cases'};

allG = [G_reported(:);G_allinf(:)];
positiveG = allG(isfinite(allG) & allG>0);

xminC = 10^floor(log10(min(positiveG)));
xmaxC = 10^ceil(log10(max(positiveG)));

xminC = max(xminC,min(positiveG)/1.7);
xmaxC = max(xmaxC,max(positiveG)*1.10);

set(axC, ...
    'XScale','log', ...
    'XLim',[xminC xmaxC], ...
    'YLim',[0.55 2.48], ...
    'YTick',yTickC, ...
    'YTickLabel',yTickLabelsC, ...
    'FontName','Arial', ...
    'FontSize',18, ...
    'LineWidth',1.0, ...
    'TickDir','out', ...
    'Box','off', ...
    'XColor',INK, ...
    'YColor',INK);

cRep = benefit;
cInf = min(1,benefit + [0.08 0.04 0.01]);

rowColors = [cRep;cInf];

rng(8);

hPtsLeg = gobjects(1);
hDenLeg = gobjects(1);
h95Leg  = gobjects(1);
h50Leg  = gobjects(1);
hMedLeg = gobjects(1);

for j = 1:2

    values = Gc{j};
    values = values(isfinite(values) & values>0);

    y0 = rowY(j);
    cRow = rowColors(j,:);

    logv = log10(values);
    logGrid = linspace(min(logv)-0.08,max(logv)+0.08,320);

    dens = kde1d_local(logv,logGrid);

    if max(dens)>0
        dens = dens/max(dens);
    end

    xGrid = 10.^logGrid;

    hDen = fill(axC, ...
        [xGrid fliplr(xGrid)], ...
        [y0 + 0.29*dens,repmat(y0,1,numel(xGrid))], ...
        cRow, ...
        'FaceAlpha',0.22, ...
        'EdgeColor',darken_local(cRow,0.25), ...
        'LineWidth',1.25);

    plot(axC,xGrid,y0+0.29*dens, ...
        'Color',darken_local(cRow,0.28), ...
        'LineWidth',1.40);

    jitter = -0.08 - 0.20*rand(numel(values),1);

    hPts = scatter(axC,values,y0+jitter,56, ...
        'MarkerFaceColor',cRow, ...
        'MarkerEdgeColor','k', ...
        'LineWidth',0.45);

    try
        hPts.MarkerFaceAlpha = 0.58;
        hPts.MarkerEdgeAlpha = 0.34;
    catch
    end

    q05 = percentile_local(values,0.05);
    q25 = percentile_local(values,0.25);
    q50 = percentile_local(values,0.50);
    q75 = percentile_local(values,0.75);
    q95 = percentile_local(values,0.95);

    ySummary = y0 + 0.045;

    h95 = plot(axC,[q05 q95],[ySummary ySummary], ...
        '-', ...
        'Color',INK, ...
        'LineWidth',1.50);

    h50 = plot(axC,[q25 q75],[ySummary ySummary], ...
        '-', ...
        'Color',INK, ...
        'LineWidth',5.2);

    hMed = plot(axC,q50,ySummary,'o', ...
        'MarkerFaceColor','w', ...
        'MarkerEdgeColor',INK, ...
        'MarkerSize',8.4, ...
        'LineWidth',1.35);

    nPositive = sum(Gc{j}>0);

    text(axC,xmaxC/1.12,y0+0.235, ...
        sprintf('%d/%d > 0   median %.1f',nPositive,Nsol,q50), ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','middle', ...
        'FontName','Arial', ...
        'FontSize',16.0, ...
        'Color',INK);

    if j == 1
        hPtsLeg = hPts;
        hDenLeg = hDen;
        h95Leg  = h95;
        h50Leg  = h50;
        hMedLeg = hMed;
    end
end

rawTicksC = [3 10 30 100 300 1000 3000 10000];
rawTicksC = rawTicksC(rawTicksC>=xminC & rawTicksC<=xmaxC);

if isempty(rawTicksC)
    rawTicksC = [min(positiveG) median(positiveG) max(positiveG)];
end

xticks(axC,rawTicksC);
xticklabels(axC,compose_ticklabels_local(rawTicksC));

xlabel(axC,'HH - LL avoided burden (10^3; logarithmic scale)', ...
    'FontSize',19, ...
    'Color',INK);

text(axC,0.02,0.96, ...
    'All values are positive: HH avoids more burden than LL in every admissible solution.', ...
    'Units','normalized', ...
    'HorizontalAlignment','left', ...
    'VerticalAlignment','top', ...
    'FontName','Arial', ...
    'FontSize',14.0, ...
    'Color',SUBINK);

lgC = legend(axC, ...
    [hPtsLeg hDenLeg h95Leg h50Leg hMedLeg], ...
    {'Admissible solutions','Density','5-95%','25-75%','Median'}, ...
    'NumColumns',3, ...
    'Box','off', ...
    'FontName','Arial', ...
    'FontSize',13.0);

lgC.Position = [0.410 0.875 0.470 0.070];

save_panel_local(figC,output_dir,'Fig4_c');

%% =========================================================================
% PANEL d — integrated 3x3 median-effect matrix
% =========================================================================
figD = figure( ...
    'Color','w', ...
    'Units','pixels', ...
    'Position',[140 100 1050 920], ...
    'Name','Fig4_d', ...
    'NumberTitle','off');

set(figD,'Renderer','painters');

annotation(figD,'textbox',[0.020 0.945 0.035 0.040], ...
    'String','d', ...
    'EdgeColor','none', ...
    'FontName','Arial', ...
    'FontSize',30, ...
    'FontWeight','bold', ...
    'Color','k');

annotation(figD,'textbox',[0.14 0.910 0.70 0.030], ...
    'String','Upper-left = GD    Lower-right = GX    Bold centre = total', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',14.0, ...
    'Color',SUBINK);

axD = axes(figD,'Position',[0.145 0.130 0.595 0.695]);
hold(axD,'on');

for r = 1:3
    for c = 1:3

        xL = c - 0.48;
        xR = c + 0.48;
        yT = r - 0.48;
        yB = r + 0.48;

        cGD = signed_color_local(Mgd(r,c),zLimD,s.signed);
        cGX = signed_color_local(Mgx(r,c),zLimD,s.signed);

        patch(axD,[xL xR xL],[yT yT yB],cGD, ...
            'EdgeColor',WHITE, ...
            'LineWidth',1.0);

        patch(axD,[xR xR xL],[yT yB yB],cGX, ...
            'EdgeColor',WHITE, ...
            'LineWidth',1.0);

        rectangle(axD, ...
            'Position',[xL yT xR-xL yB-yT], ...
            'EdgeColor',[0.70 0.72 0.72], ...
            'LineWidth',0.75);

        text(axD,c-0.22,r-0.21,format_signed_local(Mgd(r,c)), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontName','Arial', ...
            'FontSize',15, ...
            'Color',INK);

        text(axD,c+0.22,r+0.21,format_signed_local(Mgx(r,c)), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontName','Arial', ...
            'FontSize',15, ...
            'Color',INK);

        text(axD,c,r,format_signed_local(Mtot(r,c)), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','middle', ...
            'FontName','Arial', ...
            'FontSize',17, ...
            'FontWeight','bold', ...
            'Color',INK, ...
            'BackgroundColor','w', ...
            'Margin',1.1);
    end
end

set(axD, ...
    'YDir','reverse', ...
    'XLim',[0.5 3.5], ...
    'YLim',[0.5 3.5], ...
    'XTick',1:3, ...
    'YTick',1:3, ...
    'XTickLabel',{'N','H','L'}, ...
    'YTickLabel',{'N','H','L'}, ...
    'FontName','Arial', ...
    'FontSize',19, ...
    'LineWidth',1.0, ...
    'TickDir','out', ...
    'Box','off', ...
    'XColor',INK, ...
    'YColor',INK);

axis(axD,'square');

xlabel(axD,'GX strategy', ...
    'FontSize',21, ...
    'Color',INK);

ylabel(axD,'GD strategy', ...
    'FontSize',21, ...
    'Color',INK);

colormap(axD,s.signed);
clim(axD,zLimD);

cbD = colorbar(axD,'eastoutside');
cbD.Position = [0.775 0.205 0.020 0.540];

rawTicksD = choose_signed_ticks_local(limitD);

cbD.Ticks = mr.symlog(rawTicksD,1,1);
cbD.TickLabels = compose_ticklabels_local(rawTicksD);
cbD.FontName = 'Arial';
cbD.FontSize = 13;
cbD.Color = INK;

cbD.Label.String = 'Median avoided reported cases (10^3)';
cbD.Label.FontSize = 15;
cbD.Label.Color = INK;

save_panel_local(figD,output_dir,'Fig4_d');

fprintf('\nFour standalone Fig. 4 panels were saved to:\n%s\n',output_dir);

%% =========================================================================
% Local functions
% =========================================================================
function save_panel_local(fig,output_dir,name)

figFile = fullfile(output_dir,[name '.fig']);
pngFile = fullfile(output_dir,[name '.png']);

savefig(fig,figFile);

if exist(pngFile,'file')
    try
        delete(pngFile);
    catch
        pngFile = fullfile(output_dir, ...
            [name '_' datestr(now,'yyyymmdd_HHMMSS') '.png']);
    end
end

try
    exportgraphics(fig,pngFile, ...
        'Resolution',600, ...
        'BackgroundColor','white');
catch ME
    warning('exportgraphics failed for %s: %s. Falling back to print().', ...
        name,ME.message);

    print(fig,pngFile,'-dpng','-r600');
end
end


function c = signed_color_local(value,zLim,cmap)

z = mr.symlog(value,1,1);

if zLim(2) <= zLim(1)
    u = 0.5;
else
    u = (z-zLim(1))/(zLim(2)-zLim(1));
end

u = min(max(u,0),1);

idx = 1 + round(u*(size(cmap,1)-1));
idx = min(max(idx,1),size(cmap,1));

c = cmap(idx,:);
end


function out = format_signed_local(x)

if abs(x) < 0.05
    out = '0';
else
    out = sprintf('%+.1f',x);
end
end


function labels = compose_ticklabels_local(values)

labels = cell(size(values));

for i = 1:numel(values)

    v = values(i);

    if v == 0
        labels{i} = '0';
    elseif abs(v) >= 10
        labels{i} = sprintf('%.0f',v);
    elseif abs(v) >= 1
        labels{i} = sprintf('%.1f',v);
    else
        labels{i} = sprintf('%.2g',v);
    end
end
end


function ticks = choose_signed_ticks_local(limit)

preferred = [1 3 10 30 100 300 1000];

pos = preferred(preferred <= limit*1.01);

if isempty(pos)
    pos = limit;
end

if numel(pos) > 3
    pick = unique(round(linspace(1,numel(pos),3)));
    pos = pos(pick);
end

ticks = [-fliplr(pos) 0 pos];
end


function D = compute_baseline_distance_local(d)

e = d.ensemble;
b = d.baseline;

N = size(e.rates,1);
K = size(e.rates,3);

component = nan(N,K);

for k = 1:K

    vals = double(squeeze(e.rates(:,:,k)));
    base_raw = double(b.rates(:,k));

    xs = linspace(0,1,numel(base_raw));
    xt = linspace(0,1,size(vals,2));

    base_aligned = interp1(xs,base_raw,xt,'linear','extrap');

    rngk = max(double(d.bounds(k,2)-d.bounds(k,1)),eps);

    diffNorm = (vals-base_aligned)/rngk;

    component(:,k) = sqrt(mean(diffNorm.^2,2));
end

D = mean(component,2);
end


function dens = kde1d_local(x,grid)

x = x(:);
x = x(isfinite(x));

n = numel(x);

if n < 2
    dens = zeros(size(grid));
    return
end

sx = std(x);

q25 = percentile_local(x,0.25);
q75 = percentile_local(x,0.75);

iq = q75-q25;

sigma = min(sx,iq/1.349);

if ~isfinite(sigma) || sigma <= 0
    sigma = max(sx,1e-3);
end

h = max(0.9*sigma*n^(-1/5),1e-4);

dens = zeros(size(grid));

for i = 1:n
    z = (grid-x(i))/h;
    dens = dens + exp(-0.5*z.^2);
end

dens = dens/(n*h*sqrt(2*pi));
end


function q = percentile_local(x,p)

x = x(:);
x = x(isfinite(x));
x = sort(x);

if isempty(x)
    q = NaN;
    return
end

if numel(x)==1
    q = x;
    return
end

pos = 1+(numel(x)-1)*p;

lo = floor(pos);
hi = ceil(pos);

if lo == hi
    q = x(lo);
else
    a = pos-lo;
    q = (1-a)*x(lo)+a*x(hi);
end
end


function c = darken_local(c0,amount)

c = c0.*(1-amount);
c = min(max(c,0),1);
end
