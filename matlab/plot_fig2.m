function f = plot_fig2_NC_v5()
% FIGURE 2 — v5
%
% User-requested updates:
%   1) Remove the dashed independent-ODE curve from panel a.
%   2) Keep observed cases as bars and make the legend explicitly indicate
%      that they are bars.
%   3) Use larger typography:
%         - default plot/text font size ~15
%         - legend font size = 13
%
% Main editable colour blocks:
%   GD_MAIN, GX_MAIN     -> main fitted-curve colours
%   GD_OBS, GX_OBS       -> observed bar colours
%   BLUES, GREENS        -> panel b ridgeline colours
%   SPILL_FILL           -> outgoing infectious spillover fill
%   TOTAL_SPILL          -> total spillover dashed line
%   WIN_FILL             -> W1/W2/W3 background colour
%
% Notes:
%   - No title() calls because mr.save() forbids non-empty axis titles.
%   - Panel a shows bars (observed), fitted UDE line, the infectious-movement
%     spillover component as fill, and total directional spillover as a dashed line.
%   - R^2_ODE text is retained as a consistency metric, but the independent
%     ODE curve itself is not displayed.

d = mr.load_data();
s = mr.style();
b = d.baseline;

t  = b.time(:);
td = t(1:end-1) + 0.5;
WINDOWS = d.windows;

% -------------------------------------------------------------------------
% palette
% -------------------------------------------------------------------------
TXT    = [0.12 0.16 0.20];
SUBTXT = [0.36 0.39 0.41];
AXCOL  = [0.18 0.23 0.27];

GD_MAIN = [54 125 176]/255;    % #367DB0
GX_MAIN = [61 159  60]/255;    % #3D9F3C

GD_OBS = [157 199 221]/255;    % #9DC7DD
GX_OBS = [158 209 123]/255;    % #9ED17B

BLUES = [
    0.87 0.92 0.97
    0.72 0.82 0.92
    0.47 0.63 0.82
    0.21 0.44 0.69
];

GREENS = [
    0.86 0.90 0.78
    0.73 0.82 0.57
    0.53 0.66 0.36
    0.24 0.46 0.24
];

SPILL_FILL  = [0.79 0.68 0.42];
TOTAL_SPILL = [0.51 0.34 0.13];
WIN_FILL    = [0.94 0.91 0.82];

regions = {'Guangdong','Guangxi'};
reg_main = [GD_MAIN; GX_MAIN];
reg_obs  = [GD_OBS; GX_OBS];

% -------------------------------------------------------------------------
% figure canvas
% -------------------------------------------------------------------------
f = mr.newfigure('Fig2_NC_v5',10.8);
set(f,'Color','w');

% =========================================================================
% a | epidemic reconstruction and outgoing spillover context
% =========================================================================
r2ode = nan(1,2);
legend_handles = gobjects(1,4);

for j = 1:2
    y0 = 0.715 - (j-1)*0.220;
    ax = mr.axes(f,[0.095 y0 0.850 0.162]);
    hold(ax,'on');

    ymax = max([b.observed(:,j); b.differentiable(:,j); b.spillover(:,j)]);
    if ymax<=0, ymax = 1; end
    ymax = ymax * 1.14;

    % Window shading
    for k = 1:3
        w = WINDOWS(k,:);
        patch(ax,[w(1) w(2) w(2) w(1)], ...
              [0 0 ymax ymax], WIN_FILL, ...
              'FaceAlpha',0.62,'EdgeColor','none');
        text(ax,mean(w),ymax*0.91,sprintf('W%d',k), ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','top', ...
            'FontName','Arial', ...
            'FontWeight','bold', ...
            'FontSize',15, ...
            'Color',[0.44 0.41 0.35]);
    end

    % Observed bars
    hObs = bar(ax,td,b.observed(:,j),1.0, ...
        'FaceColor',reg_obs(j,:), ...
        'EdgeColor','none', ...
        'FaceAlpha',0.96);

    % Outgoing spillover as filled area + dashed total
    hSpA = area(ax,t,b.spillover_infectious(:,j), ...
        'FaceColor',SPILL_FILL, ...
        'EdgeColor','none', ...
        'FaceAlpha',0.56);

    hSpL = plot(ax,t,b.spillover(:,j),'--', ...
        'Color',TOTAL_SPILL, ...
        'LineWidth',1.75);

    % UDE fit
    hFit = plot(ax,td,b.differentiable(:,j), ...
        'Color',reg_main(j,:), ...
        'LineWidth',2.60);

    % ODE-consistency metric only (curve not shown)
    r2ode(j) = local_r2(b.observed(:,j),b.ode_daily(:,j));

    local_axis_style(ax,AXCOL,15);
    ylabel(ax,'Daily cases / spillover flow','Color',TXT,'FontSize',15);
    xlim(ax,[t(1) t(end)]);
    ylim(ax,[0 ymax]);
    xticks(ax,[0 60 120 180 236]);

    if j == 2
        xlabel(ax,'Day','Color',TXT,'FontSize',15);
    else
        set(ax,'XTickLabel',[]);
    end

    % Region label and ODE-consistency metric
    text(ax,0.012,0.90,regions{j}, ...
        'Units','normalized', ...
        'HorizontalAlignment','left', ...
        'VerticalAlignment','top', ...
        'FontName','Arial', ...
        'FontWeight','bold', ...
        'FontSize',16, ...
        'Color',reg_main(j,:));

    text(ax,0.985,0.84,sprintf('R^2_{ODE} = %.3f',r2ode(j)), ...
        'Units','normalized', ...
        'HorizontalAlignment','right', ...
        'VerticalAlignment','top', ...
        'FontName','Arial', ...
        'FontSize',15, ...
        'Color',SUBTXT);

    if j == 1
        mr.panel(ax,'a',-.14);
        legend_handles = [hObs hFit hSpA hSpL];
    end
end

lgA = legend(legend_handles, ...
    {'Observed reported cases (bars)', ...
     'UDE reconstruction', ...
     'Outgoing infectious spillover', ...
     'Total spillover'}, ...
    'NumColumns',2, ...
    'Box','off', ...
    'FontSize',13);
lgA.ItemTokenSize = [18 10];
lgA.Position = [.265 .908 .470 .050];

% =========================================================================
% b | learned time-varying functions as 4 x 2 gradient ridgelines
% =========================================================================
annotation(f,'textbox',[0.22 0.455 0.22 0.030], ...
    'String','Guangdong', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontWeight','bold', ...
    'FontSize',16, ...
    'Color',GD_MAIN);

annotation(f,'textbox',[0.58 0.455 0.22 0.030], ...
    'String','Guangxi', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontWeight','bold', ...
    'FontSize',16, ...
    'Color',GX_MAIN);

annotation(f,'textbox',[0.32 0.435 0.36 0.022], ...
    'String','Learned time-varying functions', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',15, ...
    'Color',SUBTXT);

fam_names  = {'c(t)','q(t)','m(t)','d(t)'};
fam_ranges = {'[1,16]','[0,1]','[0,0.2]','[0,0.2]'};

left_x   = 0.125;
right_x  = 0.520;
w_ax     = 0.330;
h_ax     = 0.058;
y_top    = 0.365;
row_step = 0.074;

for row = 1:4
    y0 = y_top - (row-1)*row_step;

    for j = 1:2
        if j == 1
            x0 = left_x;
            ridge_col = BLUES(row,:);
        else
            x0 = right_x;
            ridge_col = GREENS(row,:);
        end

        ax = mr.axes(f,[x0 y0 w_ax h_ax]);
        hold(ax,'on');

        k = j + 2*(row-1);
        lo = d.bounds(k,1);
        hi = d.bounds(k,2);
        z = (b.rates(:,k)-lo) ./ max(hi-lo,eps);
        z = max(0,min(1,z));

        local_gradient_ridge(ax,t,z,ridge_col);

        local_axis_style(ax,AXCOL,15);
        xlim(ax,[t(1) t(end)]);
        ylim(ax,[-0.08 1.05]);
        yticks([]);

        if row < 4
            set(ax,'XTickLabel',[]);
        else
            xticks(ax,[0 60 120 180 236]);
            xlabel(ax,'Day','Color',TXT,'FontSize',15);
        end
        ax.YColor = 'none';

        if j == 1
            text(ax,-0.055,0.72,fam_names{row}, ...
                'Units','normalized', ...
                'HorizontalAlignment','right', ...
                'VerticalAlignment','middle', ...
                'FontName','Arial', ...
                'FontWeight','bold', ...
                'FontSize',15, ...
                'Color',TXT, ...
                'Clipping','off');

            text(ax,-0.055,0.26,fam_ranges{row}, ...
                'Units','normalized', ...
                'HorizontalAlignment','right', ...
                'VerticalAlignment','middle', ...
                'FontName','Arial', ...
                'FontSize',15, ...
                'Color',SUBTXT, ...
                'Clipping','off');
        end

        if row == 1 && j == 1
            mr.panel(ax,'b',-.17);
        end
    end
end

% -------------------------------------------------------------------------
% save
% -------------------------------------------------------------------------
f.UserData.Baseline = b;
f.UserData.Windows = WINDOWS;
f.UserData.R2_independent_ODE = r2ode;

mr.save(f,'Fig2_NC_v5');

end


% =========================================================================
% local helper functions
% =========================================================================
function local_gradient_ridge(ax,x,z,colorTop)
x = x(:);
z = z(:);
nLayer = 30;

for i = 1:nLayer
    a = i / nLayer;
    c = (1-a)*[1 1 1] + a*colorTop;
    alphaVal = 0.055 + 0.028*a;
    yy = z * a;

    fill(ax,[x; flipud(x)], ...
         [zeros(size(x)); flipud(yy)], ...
         c, ...
         'EdgeColor','none', ...
         'FaceAlpha',alphaVal);
end

plot(ax,x,z,'Color',[0.13 0.13 0.13],'LineWidth',1.25);
plot(ax,[x(1) x(end)],[0 0],'Color',[0.64 0.66 0.67],'LineWidth',0.90);
end


function local_axis_style(ax,axisColor,fontSize)
if nargin < 3
    fontSize = 15;
end

set(ax, ...
    'FontName','Arial', ...
    'FontSize',fontSize, ...
    'LineWidth',0.90, ...
    'TickDir','out', ...
    'TickLength',[0.012 0.012], ...
    'Box','off', ...
    'Layer','top', ...
    'XColor',axisColor, ...
    'YColor',axisColor);
grid(ax,'off');
end


function c = lighten(c0,amount)
c = c0 + (1-c0)*amount;
c = min(max(c,0),1);
end


function r2 = local_r2(y,p)
y = y(:);
p = p(:);
mask = isfinite(y) & isfinite(p);
y = y(mask);
p = p(mask);

if isempty(y)
    r2 = NaN;
    return
end

den = sum((y - mean(y)).^2);
if den <= 0
    r2 = NaN;
else
    r2 = 1 - sum((y - p).^2) / den;
end
end
