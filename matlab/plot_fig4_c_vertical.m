%% plot_fig4_c_vertical.m
% Directly runnable MATLAB R2024b script.
%
% Vertical version of Fig. 4c:
%   x-axis: Reported cases / All infections
%   y-axis: HH - LL avoided burden (10^3, logarithmic scale)
%
% Each category shows:
%   - admissible-solution points on the left,
%   - a half-density on the right,
%   - 5-95% interval,
%   - 25-75% interval,
%   - median.
%
% Output:
%   UDEcodes\MainResults\matlab\figures\Fig4_c_vertical.fig
%   UDEcodes\MainResults\matlab\figures\Fig4_c_vertical.png

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

% HH - LL paired differences, in thousands
G_reported = (v(:,6,1,5) - v(:,9,1,5))/1000;
G_allinf   = (v(:,6,2,5) - v(:,9,2,5))/1000;

%% ------------------------------------------------------------------------
% Style
% -------------------------------------------------------------------------
INK    = [0.08 0.13 0.17];
SUBINK = [0.34 0.38 0.41];

benefit = s.benefit;

cRep = benefit;
cInf = min(1,benefit + [0.08 0.04 0.01]);
rowColors = [cRep; cInf];

%% ------------------------------------------------------------------------
% Figure canvas
% -------------------------------------------------------------------------
f = figure( ...
    'Color','w', ...
    'Units','pixels', ...
    'Position',[120 70 920 980], ...
    'Name','Fig4_c_vertical', ...
    'NumberTitle','off');

set(f,'Renderer','painters');

annotation(f,'textbox',[0.025 0.948 0.040 0.040], ...
    'String','c', ...
    'EdgeColor','none', ...
    'FontName','Arial', ...
    'FontSize',30, ...
    'FontWeight','bold', ...
    'Color','k');

ax = axes(f,'Position',[0.145 0.135 0.785 0.740]);
hold(ax,'on');

%% ------------------------------------------------------------------------
% Data preparation
% -------------------------------------------------------------------------
G = {G_reported(:),G_allinf(:)};
xCenters = [1 2];
xLabels = {'Reported cases','All infections'};

allG = [G_reported(:);G_allinf(:)];
positiveG = allG(isfinite(allG) & allG>0);

ymin = 10^floor(log10(min(positiveG)));
ymax = 10^ceil(log10(max(positiveG)));

ymin = max(ymin,min(positiveG)/1.7);
ymax = max(ymax,max(positiveG)*1.10);

set(ax, ...
    'YScale','log', ...
    'YLim',[ymin ymax], ...
    'XLim',[0.45 2.65], ...
    'XTick',xCenters, ...
    'XTickLabel',xLabels, ...
    'FontName','Arial', ...
    'FontSize',18, ...
    'LineWidth',1.0, ...
    'TickDir','out', ...
    'Box','off', ...
    'XColor',INK, ...
    'YColor',INK);

ylabel(ax,'HH - LL avoided burden (10^3; logarithmic scale)', ...
    'FontSize',19, ...
    'Color',INK);

% Human-readable y ticks
rawTicks = [3 10 30 100 300 1000 3000 10000];
rawTicks = rawTicks(rawTicks>=ymin & rawTicks<=ymax);

if isempty(rawTicks)
    rawTicks = [min(positiveG) median(positiveG) max(positiveG)];
end

yticks(ax,rawTicks);
yticklabels(ax,compose_ticklabels_local(rawTicks));

%% ------------------------------------------------------------------------
% Draw the two vertical raincloud summaries
% -------------------------------------------------------------------------
rng(8);

halfWidth = 0.28;
scatterWidth = 0.22;

hPtsLeg = gobjects(1);
hDenLeg = gobjects(1);
h95Leg  = gobjects(1);
h50Leg  = gobjects(1);
hMedLeg = gobjects(1);

for j = 1:2

    values = G{j};
    values = values(isfinite(values) & values>0);

    x0 = xCenters(j);
    cRow = rowColors(j,:);

    % Half density on the right
    logv = log10(values);
    logGrid = linspace(min(logv)-0.08,max(logv)+0.08,320);
    dens = kde1d_local(logv,logGrid);

    if max(dens)>0
        dens = dens/max(dens);
    end

    yGrid = 10.^logGrid;
    xRight = x0 + halfWidth*dens;

    hDen = fill(ax, ...
        [x0*ones(size(yGrid)) fliplr(xRight)], ...
        [yGrid fliplr(yGrid)], ...
        cRow, ...
        'FaceAlpha',0.22, ...
        'EdgeColor',darken_local(cRow,0.25), ...
        'LineWidth',1.25);

    plot(ax,xRight,yGrid, ...
        'Color',darken_local(cRow,0.28), ...
        'LineWidth',1.35);

    % Jittered points on the left
    xPts = x0 - 0.06 - scatterWidth*rand(numel(values),1);

    hPts = scatter(ax,xPts,values,54, ...
        'MarkerFaceColor',cRow, ...
        'MarkerEdgeColor','k', ...
        'LineWidth',0.45);

    try
        hPts.MarkerFaceAlpha = 0.58;
        hPts.MarkerEdgeAlpha = 0.34;
    catch
    end

    % Interval summary
    q05 = percentile_local(values,0.05);
    q25 = percentile_local(values,0.25);
    q50 = percentile_local(values,0.50);
    q75 = percentile_local(values,0.75);
    q95 = percentile_local(values,0.95);

    xSummary = x0 + 0.015;

    h95 = plot(ax,[xSummary xSummary],[q05 q95], ...
        '-', ...
        'Color',INK, ...
        'LineWidth',1.55);

    h50 = plot(ax,[xSummary xSummary],[q25 q75], ...
        '-', ...
        'Color',INK, ...
        'LineWidth',5.2);

    hMed = plot(ax,xSummary,q50,'o', ...
        'MarkerFaceColor','w', ...
        'MarkerEdgeColor',INK, ...
        'MarkerSize',8.5, ...
        'LineWidth',1.35);

    % Numerical result
    nPositive = sum(G{j}>0);

    text(ax,x0+0.31,ymax/1.35, ...
        sprintf('%d/%d > 0',nPositive,Nsol), ...
        'HorizontalAlignment','left', ...
        'VerticalAlignment','middle', ...
        'FontName','Arial', ...
        'FontSize',15.5, ...
        'Color',INK);

    text(ax,x0+0.31,ymax/2.05, ...
        sprintf('median %.1f',q50), ...
        'HorizontalAlignment','left', ...
        'VerticalAlignment','middle', ...
        'FontName','Arial', ...
        'FontSize',15.5, ...
        'Color',INK);

    if j == 1
        hPtsLeg = hPts;
        hDenLeg = hDen;
        h95Leg  = h95;
        h50Leg  = h50;
        hMedLeg = hMed;
    end
end

%% ------------------------------------------------------------------------
% Annotation and legend
% -------------------------------------------------------------------------
annotation(f,'textbox',[0.17 0.895 0.72 0.035], ...
    'String','All values are positive: HH avoids more burden than LL in every admissible solution.', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',14.5, ...
    'Color',SUBINK);

lg = legend(ax, ...
    [hPtsLeg hDenLeg h95Leg h50Leg hMedLeg], ...
    {'Admissible solutions','Density','5-95%','25-75%','Median'}, ...
    'NumColumns',2, ...
    'Box','off', ...
    'FontName','Arial', ...
    'FontSize',12.5);

lg.Position = [0.25 0.925 0.55 0.060];

%% ------------------------------------------------------------------------
% Save
% -------------------------------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));

if isempty(script_dir)
    script_dir = pwd;
end

output_dir = fullfile(script_dir,'Fig4');

if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

figFile = fullfile(output_dir,'Fig4_c_vertical.fig');
pngFile = fullfile(output_dir,'Fig4_c_vertical.png');

savefig(f,figFile);

if exist(pngFile,'file')
    try
        delete(pngFile);
    catch
        pngFile = fullfile(output_dir, ...
            ['Fig4_c_vertical_' datestr(now,'yyyymmdd_HHMMSS') '.png']);
    end
end

try
    exportgraphics(f,pngFile, ...
        'Resolution',600, ...
        'BackgroundColor','white');
catch ME
    warning('exportgraphics failed: %s. Falling back to print().',ME.message);
    print(f,pngFile,'-dpng','-r600');
end

fprintf('\nVertical Fig. 4c saved to:\n%s\n%s\n',figFile,pngFile);

%% =========================================================================
% Local functions
% =========================================================================
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
