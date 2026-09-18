%% ============================================================
% FigE3_pairwise_distance_matrix.m
% Pairwise standardized functional-distance matrix
% Schematic demonstration using the SAME ensemble as FigE2
% ============================================================

clear; clc; close all;

D = make_demo_admissible_solutions();

Dist = D.distance;
N = D.N;

%% ------------------------------------------------------------
% Figure
% ------------------------------------------------------------

fig = figure( ...
    'Color','w', ...
    'Units','centimeters', ...
    'Position',[3 3 7.0 6.2]);

ax = axes(fig);

imagesc(ax,Dist);

axis(ax,'image');

%% ------------------------------------------------------------
% Custom high-end colormap
%
% pale cream -> mint -> cyan -> blue -> deep navy
% ------------------------------------------------------------

ncol = 256;

anchors = [
    0.97 0.96 0.82
    0.82 0.93 0.77
    0.60 0.86 0.78
    0.39 0.75 0.82
    0.24 0.55 0.76
    0.17 0.35 0.64
    0.12 0.20 0.46
];

x0 = linspace(0,1,size(anchors,1));
xq = linspace(0,1,ncol);

cmap = zeros(ncol,3);

for j = 1:3
    cmap(:,j) = interp1(x0,anchors(:,j),xq,'pchip');
end

cmap = max(0,min(1,cmap));

colormap(ax,cmap);

%% ------------------------------------------------------------
% Color scaling
% ------------------------------------------------------------

% Diagonal = 0. Keep zero included.
clim(ax,[0 max(Dist(:))]);

%% ------------------------------------------------------------
% Axes
% ------------------------------------------------------------

set(ax, ...
    'FontName','Arial', ...
    'FontSize',8.8, ...
    'LineWidth',0.85, ...
    'TickDir','out', ...
    'TickLength',[0.012 0.012], ...
    'Box','on', ...
    'Layer','top');

xticks([1 20 40 60 80 100]);
yticks([1 20 40 60 80 100]);

xlabel('Solution index', ...
    'FontName','Arial', ...
    'FontSize',9.5);

ylabel('Solution index', ...
    'FontName','Arial', ...
    'FontSize',9.5);

%% ------------------------------------------------------------
% Colorbar
% ------------------------------------------------------------

cb = colorbar(ax);

cb.FontName = 'Arial';
cb.FontSize = 8.3;
cb.LineWidth = 0.7;

% Keep colorbar simple
cb.Label.String = 'Functional distance';
cb.Label.FontName = 'Arial';
cb.Label.FontSize = 9;

%% ------------------------------------------------------------
% Refined frame
% ------------------------------------------------------------

ax.XColor = [50 58 68]/255;
ax.YColor = [50 58 68]/255;

ax.Position = [0.16 0.16 0.68 0.76];

%% ------------------------------------------------------------
% Export
% ------------------------------------------------------------

exportgraphics(fig, ...
    'FigE3_pairwise_distance_matrix.pdf', ...
    'ContentType','vector');

exportgraphics(fig, ...
    'FigE3_pairwise_distance_matrix.png', ...
    'Resolution',600);

print(fig, ...
    'FigE3_pairwise_distance_matrix.svg', ...
    '-dsvg');