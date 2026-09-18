%% ============================================================
% FigE1_baseline_solution.m
% Baseline admissible solution — schematic component
% ============================================================

clear; clc; close all;

D = make_demo_admissible_solutions();

t = D.t;
y1 = D.baseline1;
y2 = D.baseline2;

%% Palette
blue      = [79 169 232]/255;       % #4FA9E8
green     = [101 201 138]/255;      % #65C98A

blueDeep  = [39 119 188]/255;
greenDeep = [48 151 95]/255;

bluePale  = [215 239 251]/255;
greenPale = [220 244 228]/255;

%% Figure
fig = figure( ...
    'Color','w', ...
    'Units','centimeters', ...
    'Position',[3 3 7.2 4.0]);

ax = axes(fig);
hold(ax,'on');

%% Soft halos
w1 = 0.035 + 0.050*y1;
w2 = 0.030 + 0.045*y2;

fill([t fliplr(t)], ...
     [y1+w1 fliplr(max(y1-w1,0))], ...
     bluePale, ...
     'EdgeColor','none', ...
     'FaceAlpha',0.70);

fill([t fliplr(t)], ...
     [y2+w2 fliplr(max(y2-w2,0))], ...
     greenPale, ...
     'EdgeColor','none', ...
     'FaceAlpha',0.68);

%% Broad under-strokes
plot(t,y1, ...
    'Color',[blue 0.24], ...
    'LineWidth',5.5);

plot(t,y2, ...
    'Color',[green 0.24], ...
    'LineWidth',5.5);

%% Main lines
plot(t,y1, ...
    'Color',blueDeep, ...
    'LineWidth',2.6);

plot(t,y2, ...
    'Color',greenDeep, ...
    'LineWidth',2.6);

%% Appearance
xlim([0 1]);
ylim([0 1.10]);

axis off;

ax.Position = [0.03 0.08 0.94 0.86];

%% Export
exportgraphics(fig, ...
    'FigE1_baseline_solution.pdf', ...
    'ContentType','vector');

exportgraphics(fig, ...
    'FigE1_baseline_solution.png', ...
    'Resolution',600);

print(fig, ...
    'FigE1_baseline_solution.svg', ...
    '-dsvg');