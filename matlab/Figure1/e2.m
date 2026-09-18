%% ============================================================
% FigE2_admissible_ensemble.m
% Diverse admissible solution ensemble — schematic component
% ============================================================

clear; clc; close all;

D = make_demo_admissible_solutions();

t    = D.t;
sol1 = D.sol1;
sol2 = D.sol2;

N = D.N;

%% ------------------------------------------------------------
% Palette
% ------------------------------------------------------------

blue      = [79 169 232]/255;
green     = [101 201 138]/255;

blueDeep  = [38 116 183]/255;
greenDeep = [45 145 91]/255;

%% ------------------------------------------------------------
% Figure
% ------------------------------------------------------------

fig = figure( ...
    'Color','w', ...
    'Units','centimeters', ...
    'Position',[3 3 9.2 4.6]);

ax = axes(fig);
hold(ax,'on');

%% ------------------------------------------------------------
% Plot all admissible solutions
% ------------------------------------------------------------

% Draw alternating order so one region does not always sit on top
order = randperm(N);

for ii = 1:N

    k = order(ii);

    % slight alpha variation creates depth
    a1 = 0.075 + 0.055*rand;
    a2 = 0.075 + 0.055*rand;

    plot(t,sol1(k,:), ...
        'Color',[blue a1], ...
        'LineWidth',1.1);

    plot(t,sol2(k,:), ...
        'Color',[green a2], ...
        'LineWidth',1.1);

end

%% ------------------------------------------------------------
% Representative ensemble trajectories
% Median is visually more robust than mean
% ------------------------------------------------------------

med1 = median(sol1,1);
med2 = median(sol2,1);

% Soft under-strokes
plot(t,med1, ...
    'Color',[blue 0.20], ...
    'LineWidth',5.0);

plot(t,med2, ...
    'Color',[green 0.20], ...
    'LineWidth',5.0);

% Main median lines
plot(t,med1, ...
    'Color',blueDeep, ...
    'LineWidth',2.0);

plot(t,med2, ...
    'Color',greenDeep, ...
    'LineWidth',2.0);

%% ------------------------------------------------------------
% Appearance
% ------------------------------------------------------------

xlim([0 1]);

ymax = max([sol1(:);sol2(:)]);
ylim([0 1.04*ymax]);

axis off;

ax.Position = [0.02 0.05 0.96 0.91];

%% ------------------------------------------------------------
% Export
% ------------------------------------------------------------

exportgraphics(fig, ...
    'FigE2_admissible_ensemble.pdf', ...
    'ContentType','vector');

exportgraphics(fig, ...
    'FigE2_admissible_ensemble.png', ...
    'Resolution',600);

print(fig, ...
    'FigE2_admissible_ensemble.svg', ...
    '-dsvg');