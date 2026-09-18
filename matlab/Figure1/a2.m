%% =========================================================
%  Panel a: Guangdong–Guangxi observed daily reported cases
%  Data source: Data_all_sm.xlsx, sheet "广东广西2"
% ==========================================================

clear; clc; close all;

%% 1. Read data
fileName  = 'Data_all_sm.xlsx';
sheetName = '广东广西2';

T = readtable(fileName, ...
    'Sheet', sheetName, ...
    'VariableNamingRule', 'preserve');

date = T.DATE;
GD   = T.GD;
GX   = T.GX;

% Make sure date is datetime
if ~isdatetime(date)
    date = datetime(date);
end

%% 2. Nature-like muted palette
% Guangdong: dusty blue
blue = [79, 169, 232] / 255;       % #4FA9E8

% Guangxi: soft sage green
green = [101, 201, 138] / 255;     % #65C98A

% Axis/text colour
axisColor = [55, 62, 70] / 255;

%% 3. Create figure
fig = figure( ...
    'Color', 'w', ...
    'Units', 'centimeters', ...
    'Position', [3 3 15.5 5.7]);

ax = axes(fig);
hold(ax, 'on');

%% 4. Plot observed daily cases
%
% Both series are placed at the true date locations.
% Transparency lets overlapping bars remain visible.

bGD = bar(ax, date, GD, 0.75, ...
    'FaceColor', blue, ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.88);

bGX = bar(ax, date, GX, 0.52, ...
    'FaceColor', green, ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.9);

%% 5. Axis formatting
xlim(ax, [date(1)-days(2), date(end)+days(2)]);

% Start y-axis from zero
ylim(ax, [0, 1.08 * max([GD; GX])]);

ylabel(ax, 'Daily reported cases', ...
    'FontName', 'Arial', ...
    'FontSize', 9);

% Monthly ticks
tickDates = dateshift(date(1), 'start', 'month') : calmonths(1) : ...
            dateshift(date(end), 'start', 'month');

% Keep ticks inside actual data interval
tickDates = tickDates(tickDates >= date(1) & tickDates <= date(end));

xticks(ax, tickDates);
xtickformat(ax, 'yyyy/MM');

%% 6. Publication-style appearance
set(ax, ...
    'FontName', 'Arial', ...
    'FontSize', 8.5, ...
    'LineWidth', 0.8, ...
    'TickDir', 'out', ...
    'TickLength', [0.015 0.015], ...
    'XColor', axisColor, ...
    'YColor', axisColor, ...
    'Box', 'off', ...
    'Layer', 'top');

% No grid for a cleaner Nature-like panel
grid(ax, 'off');

% Remove excessive whitespace
ax.LooseInset = max(ax.TightInset, 0.02);

%% 7. Legend
lgd = legend(ax, [bGD, bGX], ...
    {'Guangdong', 'Guangxi'}, ...
    'Location', 'northeast', ...
    'Box', 'off', ...
    'FontName', 'Arial', ...
    'FontSize', 8);

lgd.ItemTokenSize = [13, 8];

%% 8. Optional: slightly lighter background
ax.Color = [1 1 1];

%% 9. Export
% Vector PDF: best for manuscript assembly
exportgraphics(fig, 'PanelA_GD_GX_daily_cases.pdf', ...
    'ContentType', 'vector');

% High-resolution PNG: useful for PowerPoint / Illustrator preview
exportgraphics(fig, 'PanelA_GD_GX_daily_cases.png', ...
    'Resolution', 600);