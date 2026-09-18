%% ============================================================
%  Panel d — Baseline admissible reconstruction
%  Refined conceptual illustration
%
%  NOTE:
%  - Synthetic data only
%  - Shaded ribbons are graphical halos, NOT confidence intervals
%  - Designed for a Nature Communications-style workflow figure
% ============================================================

clear; clc; close all;
rng(2026);

%% ============================================================
% 1. Synthetic time axis
% ============================================================
t     = linspace(0,100,601);
t_obs = linspace(2,98,42);

%% ============================================================
% 2. Construct two smooth, asymmetric schematic trajectories
% ============================================================

% ----- Region 1 -----
f1 = ...
    0.76 * exp(-0.5*((t-41)/11.5).^2) + ...
    0.20 * exp(-0.5*((t-56)/18).^2) + ...
    0.055* exp(-0.5*((t-20)/9).^2);

% slight asymmetric modulation
f1 = f1 .* (1 + 0.055*tanh((t-38)/15));
f1 = f1 + 0.012;

% ----- Region 2 -----
f2 = ...
    0.39 * exp(-0.5*((t-37)/13.5).^2) + ...
    0.115* exp(-0.5*((t-56)/20).^2) + ...
    0.028* exp(-0.5*((t-17)/8).^2);

f2 = f2 .* (1 + 0.045*tanh((t-34)/17));
f2 = f2 + 0.008;

%% ============================================================
% 3. Synthetic observations
% ============================================================
f1o = interp1(t,f1,t_obs,'pchip');
f2o = interp1(t,f2,t_obs,'pchip');

% Heteroscedastic observation noise
sd1 = 0.020 + 0.060*f1o;
sd2 = 0.016 + 0.065*f2o;

obs1 = max(0, f1o + sd1.*randn(size(t_obs)));
obs2 = max(0, f2o + sd2.*randn(size(t_obs)));

%% ============================================================
% 4. Refined palette
% ============================================================

% Main bright colors
blue      = [79 169 232]/255;       % #4FA9E8
green     = [101 201 138]/255;      % #65C98A

% Deep line colors
blueDeep  = [38 116 183]/255;       % refined blue
greenDeep = [45 145 91]/255;        % refined green

% Very pale halo colors
bluePale  = [214 238 251]/255;
greenPale = [220 244 228]/255;

% Ghost trajectory colors
blueGhost  = [142 204 240]/255;
greenGhost = [151 220 174]/255;

% Neutral colors
axisCol = [45 52 62]/255;
textCol = [39 48 60]/255;

%% ============================================================
% 5. Construct narrow graphical halos
%    These are NOT confidence intervals
% ============================================================

% Width follows the magnitude of each curve
w1 = 0.020 + 0.070*f1;
w2 = 0.016 + 0.060*f2;

upper1 = f1 + w1;
lower1 = max(0,f1 - w1);

upper2 = f2 + w2;
lower2 = max(0,f2 - w2);

%% ============================================================
% 6. Figure
% ============================================================

fig = figure( ...
    'Color','w', ...
    'Units','centimeters', ...
    'Position',[3 3 14.0 6.6]);

ax = axes(fig);
hold(ax,'on');

%% ============================================================
% 7. Very subtle halos
% ============================================================

hHalo1 = fill( ...
    [t fliplr(t)], ...
    [upper1 fliplr(lower1)], ...
    bluePale, ...
    'EdgeColor','none', ...
    'FaceAlpha',0.58);

hHalo2 = fill( ...
    [t fliplr(t)], ...
    [upper2 fliplr(lower2)], ...
    greenPale, ...
    'EdgeColor','none', ...
    'FaceAlpha',0.58);

%% ============================================================
% 8. Ghost trajectories
%    Slightly displaced only for visual depth
% ============================================================

ghost1 = f1 .* (0.975 + 0.018*sin(2*pi*t/72));
ghost2 = f2 .* (1.020 - 0.020*sin(2*pi*t/67));

plot(t,ghost1, ...
    'Color',[blueGhost 0.48], ...
    'LineWidth',1.05);

plot(t,ghost2, ...
    'Color',[greenGhost 0.48], ...
    'LineWidth',1.05);

%% ============================================================
% 9. Main fitted curves
%
% Double-layer line:
% pale broad under-stroke + sharp dark foreground
% ============================================================

% Region 1 under-stroke
plot(t,f1, ...
    'Color',[blue 0.30], ...
    'LineWidth',5.0);

% Region 1 main line
p1 = plot(t,f1, ...
    'Color',blueDeep, ...
    'LineWidth',2.35);

% Region 2 under-stroke
plot(t,f2, ...
    'Color',[green 0.30], ...
    'LineWidth',5.0);

% Region 2 main line
p2 = plot(t,f2, ...
    'Color',greenDeep, ...
    'LineWidth',2.35);

%% ============================================================
% 10. Observed points
% ============================================================

% Slightly varying point size makes schematic data feel organic
size1 = 43 + 15*rand(size(t_obs));
size2 = 43 + 15*rand(size(t_obs));

s1 = scatter( ...
    t_obs,obs1,size1, ...
    'o', ...
    'MarkerFaceColor',blue, ...
    'MarkerEdgeColor','w', ...
    'LineWidth',0.75, ...
    'MarkerFaceAlpha',0.82);

s2 = scatter( ...
    t_obs,obs2,size2, ...
    'o', ...
    'MarkerFaceColor',green, ...
    'MarkerEdgeColor','w', ...
    'LineWidth',0.75, ...
    'MarkerFaceAlpha',0.82);

%% ============================================================
% 11. Replot main lines above observations
%     Gives a very crisp final hierarchy
% ============================================================

p1 = plot(t,f1, ...
    'Color',blueDeep, ...
    'LineWidth',2.4);

p2 = plot(t,f2, ...
    'Color',greenDeep, ...
    'LineWidth',2.4);

%% ============================================================
% 12. Axes
% ============================================================

xlim([0 100]);
ylim([0 1.08]);

xlabel('Time', ...
    'FontName','Arial', ...
    'FontSize',10.5, ...
    'Color',textCol);

ylabel('Daily cases', ...
    'FontName','Arial', ...
    'FontSize',10.5, ...
    'Color',textCol);

set(ax, ...
    'FontName','Arial', ...
    'FontSize',15, ...
    'LineWidth',0.95, ...
    'TickDir','out', ...
    'TickLength',[0.015 0.015], ...
    'XColor',axisCol, ...
    'YColor',axisCol, ...
    'Box','off', ...
    'Layer','top');

% Conceptual schematic: no numerical ticks
ax.XTick = [];
ax.YTick = [];

grid(ax,'off');

%% ============================================================
% 13. Legend
% ============================================================

lgd = legend( ...
    [s1 p1 s2 p2], ...
    {'Observed (region 1)', ...
     'Fit (region 1)', ...
     'Observed (region 2)', ...
     'Fit (region 2)'}, ...
    'Location','northeast', ...
    'Box','off', ...
    'FontName','Arial', ...
    'FontSize',14);

lgd.ItemTokenSize = [17 8];

%% ============================================================
% 14. Layout
% ============================================================

ax.Position = [0.105 0.16 0.85 0.79];

%% ============================================================
% 15. Export
% ============================================================

% exportgraphics(fig, ...
%     'PanelD_baseline_reconstruction_refined.pdf', ...
%     'ContentType','vector');
% 
% exportgraphics(fig, ...
%     'PanelD_baseline_reconstruction_refined.png', ...
%     'Resolution',600);
% 
% print(fig, ...
%     'PanelD_baseline_reconstruction_refined.svg', ...
%     '-dsvg');