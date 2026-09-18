function out = plot_fig3_NC_v2_fix4_densityfix(output_dir)
% FIGURE 3 — v2 fix3
%
% This version removes ALL tiledlayout/nexttile usage so every axes object
% remains a normal editable MATLAB axes in the saved .fig file.
%
% Outputs:
%   Fig3_a.fig / Fig3_a.png
%   Fig3_b.fig / Fig3_b.png
%   Fig3_c.fig / Fig3_c.png
%   Fig3_d.fig / Fig3_d.png
%
% Panel A:
%   4 x 2 manually positioned axes.
%   Left  = Guangdong: c1, q1, m1, d1
%   Right = Guangxi:   c2, q2, m2, d2
%   Both columns retain their y-axis ticks and labels.
%
% Panel B:
%   2 x 4 scatter + marginal-density panels for baseline-relative differences.
%
% Panel C:
%   Full 100 x 100 pairwise-distance matrix + distance distribution.
%
% Panel D:
%   Two directional spillover ribbon summaries.
%
% Only .fig and .png are exported.

if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(pwd,'Fig3_panels');
end
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

d = mr.load_data();
e = d.ensemble;
b = d.baseline;
Dmat = double(d.distances);

t = double(e.time(:));
N = size(e.rates,1);

% Robust plotting grid for baseline arrays; avoids dependence on b.time type.
base_t = linspace(t(1),t(end),size(b.rates,1))';

% -------------------------------------------------------------------------
% typography + palette
% -------------------------------------------------------------------------
FS    = 15;
FS_LG = 17;
FS_SM = 13;

TXT    = [0.12 0.16 0.20];
SUBTXT = [0.38 0.40 0.43];
AXCOL  = [0.20 0.24 0.28];

GD    = [54 125 176]/255;
GX    = [61 159  60]/255;
GD_L1 = [214 229 241]/255;
GD_L2 = [157 199 221]/255;
GX_L1 = [227 239 214]/255;
GX_L2 = [158 209 123]/255;

DIST_CMAP = make_distance_cmap(256);

% Raw storage:
% [c1,c2,q1,q2,m1,m2,d1,d2]
func_labels_raw = {'c_1(t)','c_2(t)','q_1(t)','q_2(t)', ...
                   'm_1(t)','m_2(t)','d_1(t)','d_2(t)'};

% Desired display order:
% row-wise for Panel A: [c1,c2], [q1,q2], [m1,m2], [d1,d2]
panelA_idx = [1 2; 3 4; 5 6; 7 8];

% Panel B:
% top row = GD c1 q1 m1 d1
% bottom row = GX c2 q2 m2 d2
panelB_idx = [1 3 5 7 2 4 6 8];

% -------------------------------------------------------------------------
% baseline-relative functional distances (used for representative indices)
% -------------------------------------------------------------------------
dist_to_base = nan(N,8);

for k = 1:8
    vals = double(squeeze(e.rates(:,:,k)));  % N x T

    base_raw = double(b.rates(:,k));
    x_src = linspace(0,1,numel(base_raw));
    x_tgt = linspace(0,1,size(vals,2));
    base_aligned = interp1(x_src,base_raw,x_tgt,'linear','extrap');

    rngk = max(double(d.bounds(k,2)-d.bounds(k,1)),eps);
    dist_to_base(:,k) = mean(abs(vals-base_aligned),2)/rngk;
end

overall_dist = mean(dist_to_base,2);
rep_idx = representative_indices(overall_dist,8);

% =========================================================================
% PANEL A — 8 ribbon summaries, manually positioned
% =========================================================================
figA = figure('Color','w','Units','pixels','Position',[80 50 1320 1180]);

% Normal axes positions -> editable/draggable in .fig
xPos = [0.09 0.55];
axW  = 0.38;
axH  = 0.145;
yPos = [0.72 0.52 0.32 0.12];

% panel label
annotation(figA,'textbox',[0.018 0.942 0.035 0.04], ...
    'String','a','EdgeColor','none','FontName','Arial', ...
    'FontSize',28,'FontWeight','bold','Color','k');

% column headers safely inside figure
annotation(figA,'textbox',[0.17 0.895 0.22 0.035], ...
    'String','Guangdong','EdgeColor','none', ...
    'HorizontalAlignment','center','FontName','Arial', ...
    'FontSize',FS_LG,'FontWeight','bold','Color',GD);

annotation(figA,'textbox',[0.63 0.895 0.22 0.035], ...
    'String','Guangxi','EdgeColor','none', ...
    'HorizontalAlignment','center','FontName','Arial', ...
    'FontSize',FS_LG,'FontWeight','bold','Color',GX);

% compact descriptive line, below legend zone
annotation(figA,'textbox',[0.30 0.865 0.40 0.025], ...
    'String','Bands summarize the 100 admissible solutions; solid line denotes the baseline', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName','Arial','FontSize',FS_SM,'Color',SUBTXT);

firstAx = [];
legHandles = gobjects(1,5);

for row = 1:4
    for col = 1:2
        k = panelA_idx(row,col);

        ax = axes(figA,'Position',[xPos(col) yPos(row) axW axH]);
        hold(ax,'on');

        if isempty(firstAx)
            firstAx = ax;
        end

        vals = double(squeeze(e.rates(:,:,k)));

        q05 = quantile_curve(vals,0.05);
        q25 = quantile_curve(vals,0.25);
        q50 = quantile_curve(vals,0.50);
        q75 = quantile_curve(vals,0.75);
        q95 = quantile_curve(vals,0.95);

        if col == 1
            mainC = GD; c95 = GD_L1; c50 = GD_L2;
        else
            mainC = GX; c95 = GX_L1; c50 = GX_L2;
        end

        reps = gobjects(numel(rep_idx),1);
        for ii = 1:numel(rep_idx)
            reps(ii) = plot(ax,t,vals(rep_idx(ii),:), ...
                'Color',lighten(mainC,0.73),'LineWidth',0.8);
        end

        h95 = fill(ax,[t;flipud(t)],[q05';flipud(q95')],c95, ...
            'EdgeColor','none','FaceAlpha',0.92);
        h50 = fill(ax,[t;flipud(t)],[q25';flipud(q75')],c50, ...
            'EdgeColor','none','FaceAlpha',0.92);
        hMed = plot(ax,t,q50,'--', ...
            'Color',darken(mainC,0.25),'LineWidth',1.5);

        hBase = plot(ax,base_t,b.rates(:,k),'-', ...
            'Color',mainC,'LineWidth',2.6);

        bounds = double(d.bounds(k,:));
        xlim(ax,[t(1) t(end)]);
        ylim(ax,bounds);
        xticks(ax,[0 60 120 180 236]);
        yticks(ax,linspace(bounds(1),bounds(2),3));
        nice_axes(ax,AXCOL,FS);

        % BOTH columns retain y ticks and parameter labels.
        ylabel(ax,func_labels_raw{k}, ...
            'Interpreter','tex','FontSize',FS,'Color',TXT);

        if row < 4
            set(ax,'XTickLabel',[]);
        else
            xlabel(ax,'Day','FontSize',FS,'Color',TXT);
        end

        if row==1 && col==1
            legHandles = [reps(1) h95 h50 hMed hBase];
        end
    end
end

% Legend attached to a normal axes, not the layout.
lgA = legend(firstAx,legHandles, ...
    {'Representative solutions','5–95% band','25–75% band', ...
     'Ensemble median','Baseline'}, ...
    'NumColumns',3,'Box','off','FontSize',FS_SM);
lgA.Position = [0.245 0.935 0.53 0.050];

save_panel(figA,output_dir,'Fig3_a');

% =========================================================================
% PANEL B — 4 rows × 2 columns scatter + marginal-density panels
% =========================================================================
figB = figure( ...
    'Color','w', ...
    'Units','pixels', ...
    'Position',[80 25 1320 1480]);

% -------------------------------------------------------------------------
% panel label / column headers
% -------------------------------------------------------------------------
annotation(figB,'textbox',[0.018 0.958 0.035 0.035], ...
    'String','b', ...
    'EdgeColor','none', ...
    'FontName','Arial', ...
    'FontSize',20, ...
    'FontWeight','bold', ...
    'Color','k');

annotation(figB,'textbox',[0.17 0.942 0.25 0.032], ...
    'String','Guangdong', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',16, ...
    'FontWeight','bold', ...
    'Color',GD);

annotation(figB,'textbox',[0.58 0.942 0.25 0.032], ...
    'String','Guangxi', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',16, ...
    'FontWeight','bold', ...
    'Color',GX);

annotation(figB,'textbox',[0.22 0.974 0.56 0.020], ...
    'String','Baseline-relative differences across 100 admissible solutions', ...
    'EdgeColor','none', ...
    'HorizontalAlignment','center', ...
    'FontName','Arial', ...
    'FontSize',12, ...
    'Color',SUBTXT);


% =========================================================================
% Compute descriptive coordinates
%
% x = normalized mean shift relative to baseline
% y = normalized roughness shift relative to baseline
% =========================================================================
X  = nan(N,8);
Y  = nan(N,8);
Db = nan(N,8);

for k = 1:8

    vals = double(squeeze(e.rates(:,:,k)));    % N × T
    base_raw = double(b.rates(:,k));

    % Align baseline to ensemble temporal grid
    xs = linspace(0,1,numel(base_raw));
    xt = linspace(0,1,size(vals,2));

    base_aligned = interp1( ...
        xs, ...
        base_raw, ...
        xt, ...
        'linear', ...
        'extrap');

    rngk = max( ...
        double(d.bounds(k,2)-d.bounds(k,1)), ...
        eps);

    % normalized mean shift
    X(:,k) = ...
        mean(vals-base_aligned,2) ./ rngk;

    % normalized roughness shift
    tv_base = mean(abs(diff(base_aligned)));

    tv_vals = ...
        mean(abs(diff(vals,1,2)),2);

    Y(:,k) = ...
        (tv_vals-tv_base) ./ rngk;

    % baseline-relative component functional distance.  This uses the
    % same normalized piecewise-linear RMS definition as the manuscript
    % functional distance D, before averaging across the eight functions.
    diffNorm = (vals-base_aligned) ./ rngk;
    dtLocal = (t(end)-t(1)) / max(size(vals,2)-1,1);
    TLocal = max(t(end)-t(1),eps);
    Db(:,k) = sqrt(sum((diffNorm(:,1:end-1).^2 + ...
                        diffNorm(:,1:end-1).*diffNorm(:,2:end) + ...
                        diffNorm(:,2:end).^2) * (dtLocal/3),2) / TLocal);

end


% =========================================================================
% Layout / display order
%
% raw storage:
% [c1 c2 q1 q2 m1 m2 d1 d2]
%
% display:
% row1: c1 | c2
% row2: q1 | q2
% row3: m1 | m2
% row4: d1 | d2
% =========================================================================
panelB_idx = [
    1 2
    3 4
    5 6
    7 8
];


% -------------------------------------------------------------------------
% High-end colour palette
% -------------------------------------------------------------------------
PANELB_SCATTER_COLS = zeros(4,2,3);

% c
PANELB_SCATTER_COLS(1,1,:) = [0.79 0.61 0.25];   % c1 ochre
PANELB_SCATTER_COLS(1,2,:) = [0.56 0.69 0.38];   % c2 sage

% q
PANELB_SCATTER_COLS(2,1,:) = [0.72 0.42 0.33];   % q1 coral-brown
PANELB_SCATTER_COLS(2,2,:) = [0.70 0.50 0.45];   % q2 rose-brown

% m
PANELB_SCATTER_COLS(3,1,:) = [0.41 0.60 0.33];   % m1 leaf green
PANELB_SCATTER_COLS(3,2,:) = [0.48 0.57 0.77];   % m2 periwinkle

% d
PANELB_SCATTER_COLS(4,1,:) = [0.38 0.53 0.72];   % d1 steel blue
PANELB_SCATTER_COLS(4,2,:) = [0.62 0.52 0.70];   % d2 mauve


% =========================================================================
% Determine row-specific limits
%
% Same parameter family shares the same axes between GD and GX,
% but c/q/m/d are allowed to use different scales.
% =========================================================================
rowXLim = zeros(4,2);
rowYLim = zeros(4,2);

for row = 1:4

    ids = panelB_idx(row,:);

    xx = X(:,ids);
    yy = Y(:,ids);

    xx = xx(isfinite(xx));
    yy = yy(isfinite(yy));

    % robust limits
    xp = prctile(xx,[2 98]);
    yp = prctile(yy,[2 98]);

    xM = max(abs(xp));
    yM = max(abs(yp));

    % padding
    xM = max(xM*1.22, 1e-4);
    yM = max(yM*1.22, 1e-4);

    % symmetric around baseline = 0
    rowXLim(row,:) = [-xM xM];
    rowYLim(row,:) = [-yM yM];

end


% =========================================================================
% Manual 4 × 2 layout
% =========================================================================
groupX = [ ...
    0.105 ...    % Guangdong
    0.565 ...    % Guangxi
];

groupY = [ ...
    0.755 ...    % c
    0.535 ...    % q
    0.315 ...    % m
    0.095 ...    % d
];

centerW = 0.295;
centerH = 0.128;

topH   = 0.040;
rightW = 0.055;

gapTop  = 0.000;
gapSide = 0.004;


% =========================================================================
% Draw 8 joint plots
% =========================================================================
for row = 1:4

    for col = 1:2

        k = panelB_idx(row,col);

        c = squeeze( ...
            PANELB_SCATTER_COLS(row,col,:) ...
            )';

        gx = groupX(col);
        gy = groupY(row);

        xLim = rowXLim(row,:);
        yLim = rowYLim(row,:);


        % ================================================================
        % Central scatter
        % ================================================================
        ax = axes( ...
            figB, ...
            'Position',[ ...
            gx ...
            gy ...
            centerW ...
            centerH]);

        hold(ax,'on');

        nice_axes(ax,[0 0 0],17);

        xlim(ax,xLim);
        ylim(ax,yLim);

        % ---------------------------------------------------------------
        % baseline crosshair
        % ---------------------------------------------------------------
        plot(ax, ...
            [0 0], ...
            yLim, ...
            '--', ...
            'Color','k', ...
            'LineWidth',1.05);

        plot(ax, ...
            xLim, ...
            [0 0], ...
            '--', ...
            'Color','k', ...
            'LineWidth',1.05);


        % ---------------------------------------------------------------
        % scatter cloud
        % ---------------------------------------------------------------
        s = scatter( ...
            ax, ...
            X(:,k), ...
            Y(:,k), ...
            52, ...
            'MarkerFaceColor',c, ...
            'MarkerEdgeColor','k', ...
            'LineWidth',0.52);

        try
            s.MarkerFaceAlpha = 0.44;
            s.MarkerEdgeAlpha = 0.38;
        catch
        end


        % ---------------------------------------------------------------
        % descriptive regression line
        % ---------------------------------------------------------------
        ok = ...
            isfinite(X(:,k)) & ...
            isfinite(Y(:,k));

        if nnz(ok) >= 3

            p = polyfit( ...
                X(ok,k), ...
                Y(ok,k), ...
                1);

            xx = linspace( ...
                xLim(1), ...
                xLim(2), ...
                200);

            yy = polyval(p,xx);

            plot( ...
                ax, ...
                xx, ...
                yy, ...
                '-', ...
                'Color',darken(c,0.18), ...
                'LineWidth',2.3);

        end


        % ---------------------------------------------------------------
        % baseline origin
        % ---------------------------------------------------------------
        plot( ...
            ax, ...
            0, ...
            0, ...
            'p', ...
            'MarkerSize',7, ...
            'MarkerFaceColor',[0.20 0.20 0.20], ...
            'MarkerEdgeColor','w', ...
            'LineWidth',0.9);


        % ---------------------------------------------------------------
        % annotations
        % ---------------------------------------------------------------
        text( ...
            ax, ...
            0.035, ...
            0.94, ...
            sprintf('n = %d',N), ...
            'Units','normalized', ...
            'HorizontalAlignment','left', ...
            'VerticalAlignment','top', ...
            'FontName','Arial', ...
            'FontSize',14, ...
            'Color',SUBTXT);


        medD = median( ...
            Db(:,k), ...
            'omitnan');


        text( ...
            ax, ...
            0.965, ...
            0.055, ...
            sprintf('med D = %.3f',medD), ...
            'Units','normalized', ...
            'HorizontalAlignment','right', ...
            'VerticalAlignment','bottom', ...
            'FontName','Arial', ...
            'FontSize',12, ...
            'Color',SUBTXT);


        % ---------------------------------------------------------------
        % Function label
        %
        % Use figure annotation instead of MATLAB title() so that the
        % marginal density above does not cover it.
        % ---------------------------------------------------------------
        annotation( ...
            figB, ...
            'textbox', ...
            [ ...
            gx ...
            gy + centerH + topH + 0.002 ...
            centerW ...
            0.026 ...
            ], ...
            'String',func_labels_raw{k}, ...
            'Interpreter','tex', ...
            'EdgeColor','none', ...
            'HorizontalAlignment','center', ...
            'FontName','Arial', ...
            'FontSize',14, ...
            'FontWeight','bold', ...
            'Color',darken(c,0.28));


        % ---------------------------------------------------------------
        % axes labels
        % ---------------------------------------------------------------
        if col == 1

            ylabel( ...
                ax, ...
                'Roughness shift', ...
                'FontSize',12, ...
                'Color',TXT);

        else

            % Keep ticks because each function family has its own scale,
            % while GD/GX of the same row share the same limits.
            ylabel( ...
                ax, ...
                'Roughness shift', ...
                'FontSize',12, ...
                'Color',TXT);

        end


        if row == 4

            xlabel( ...
                ax, ...
                'Mean shift', ...
                'FontSize',12, ...
                'Color',TXT);

        else

            set( ...
                ax, ...
                'XTickLabel',[]);

        end


        % ================================================================
        % Top marginal density
        % ================================================================
        axTop = axes( ...
            figB, ...
            'Position',[ ...
            gx ...
            gy + centerH + gapTop ...
            centerW ...
            topH]);

        hold(axTop,'on');

        draw_top_density_B( ...
            axTop, ...
            X(:,k), ...
            xLim, ...
            c);


        % ================================================================
        % Right marginal density
        % ================================================================
        axR = axes( ...
            figB, ...
            'Position',[ ...
            gx + centerW + gapSide ...
            gy ...
            rightW ...
            centerH]);

        hold(axR,'on');

        draw_right_density_B( ...
            axR, ...
            Y(:,k), ...
            yLim, ...
            c);

    end

end


% =========================================================================
% Export
% =========================================================================
save_panel( ...
    figB, ...
    output_dir, ...
    'Fig3_b');

% =========================================================================
% PANEL C — full matrix + distribution
% =========================================================================
figC = figure('Color','w','Units','pixels','Position',[160 110 1340 770]);

annotation(figC,'textbox',[0.018 0.935 0.035 0.04], ...
    'String','c','EdgeColor','none','FontName','Arial', ...
    'FontSize',28,'FontWeight','bold','Color','k');

annotation(figC,'textbox',[0.28 0.925 0.44 0.03], ...
    'String','Global diversity among the 100 admissible solutions', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName','Arial','FontSize',FS_SM,'Color',SUBTXT);

axC1 = axes(figC,'Position',[0.08 0.18 0.43 0.66]);
imagesc(axC1,Dmat);
axis(axC1,'image');
set(axC1,'YDir','reverse');
colormap(axC1,DIST_CMAP);
caxis(axC1,[0 max(Dmat(:))]);

xlabel(axC1,'Solution','FontSize',FS,'Color',TXT);
ylabel(axC1,'Solution','FontSize',FS,'Color',TXT);
xticks(axC1,[1 50 100]);
yticks(axC1,[1 50 100]);
nice_axes(axC1,AXCOL,FS);

cb = colorbar(axC1,'southoutside');
cb.Position = [0.09 0.085 0.41 0.024];
cb.FontName = 'Arial';
cb.FontSize = FS_SM;
cb.Label.String = 'Functional distance D';
cb.Label.FontSize = FS_SM;

values = Dmat(triu(true(size(Dmat)),1));
values = values(isfinite(values));
vmin = min(values); vmed = median(values); vmax = max(values);

axC2 = axes(figC,'Position',[0.61 0.22 0.32 0.55]);
hold(axC2,'on');

edges = linspace(min(values),max(values),22);
histogram(axC2,values,edges, ...
    'FaceColor',[0.42 0.72 0.75], ...
    'EdgeColor','w','FaceAlpha',0.90);

yl = ylim(axC2);
xg = linspace(min(values),max(values),300);
dens = kde1d(values,xg);
if max(dens)>0
    dens = dens/max(dens)*(0.90*yl(2));
end
plot(axC2,xg,dens,'Color',[0.14 0.36 0.44],'LineWidth',2.0);

yl = ylim(axC2);
plot(axC2,[vmin vmin],[0 yl(2)*0.78],'--', ...
    'Color',[0.78 0.57 0.24],'LineWidth',1.5);
plot(axC2,[vmed vmed],[0 yl(2)*0.92],'-', ...
    'Color',[0.10 0.21 0.27],'LineWidth',2.1);
plot(axC2,[vmax vmax],[0 yl(2)*0.72],'--', ...
    'Color',[0.78 0.57 0.24],'LineWidth',1.5);

text(axC2,vmin,yl(2)*0.82,sprintf('min %.3f',vmin), ...
    'HorizontalAlignment','center','FontSize',FS_SM,'Color',SUBTXT);
text(axC2,vmed,yl(2)*0.96,sprintf('median %.3f',vmed), ...
    'HorizontalAlignment','center','FontSize',FS_SM, ...
    'FontWeight','bold','Color',TXT);
text(axC2,vmax,yl(2)*0.76,sprintf('max %.3f',vmax), ...
    'HorizontalAlignment','center','FontSize',FS_SM,'Color',SUBTXT);
text(axC2,0.98,0.98,sprintf('%d pairs',numel(values)), ...
    'Units','normalized','HorizontalAlignment','right', ...
    'VerticalAlignment','top','FontSize',FS_SM,'Color',SUBTXT);

xlabel(axC2,'Functional distance D','FontSize',FS,'Color',TXT);
ylabel(axC2,'Pairs','FontSize',FS,'Color',TXT);
nice_axes(axC2,AXCOL,FS);

save_panel(figC,output_dir,'Fig3_c');

% =========================================================================
% PANEL D — directional spillover ribbon summaries, manual axes
% =========================================================================
figD = figure('Color','w','Units','pixels','Position',[210 130 1280 720]);

annotation(figD,'textbox',[0.018 0.935 0.035 0.04], ...
    'String','d','EdgeColor','none','FontName','Arial', ...
    'FontSize',28,'FontWeight','bold','Color','k');

annotation(figD,'textbox',[0.28 0.935 0.44 0.03], ...
    'String','Directional spillover across the admissible solution ensemble', ...
    'EdgeColor','none','HorizontalAlignment','center', ...
    'FontName','Arial','FontSize',FS_SM,'Color',SUBTXT);

dirLabs = {'GD \rightarrow GX','GX \rightarrow GD'};
dirCols = [GD;GX];
dirL1 = [GD_L1;GX_L1];
dirL2 = [GD_L2;GX_L2];

dY = [0.55 0.12];
dH = 0.29;
firstD = [];
legD = gobjects(1,5);

% baseline spillover plotting grid
base_flux_t = linspace(t(1),t(end),size(b.spillover,1))';

for j = 1:2
    ax = axes(figD,'Position',[0.10 dY(j) 0.84 dH]);
    hold(ax,'on');

    if isempty(firstD), firstD = ax; end

    vals = double(squeeze(e.fluxes(:,:,j)));

    q05 = quantile_curve(vals,0.05);
    q25 = quantile_curve(vals,0.25);
    q50 = quantile_curve(vals,0.50);
    q75 = quantile_curve(vals,0.75);
    q95 = quantile_curve(vals,0.95);

    reps = gobjects(numel(rep_idx),1);
    for ii = 1:numel(rep_idx)
        reps(ii) = plot(ax,t,vals(rep_idx(ii),:), ...
            'Color',lighten(dirCols(j,:),0.75),'LineWidth',0.8);
    end

    h95 = fill(ax,[t;flipud(t)],[q05';flipud(q95')],dirL1(j,:), ...
        'EdgeColor','none','FaceAlpha',0.92);
    h50 = fill(ax,[t;flipud(t)],[q25';flipud(q75')],dirL2(j,:), ...
        'EdgeColor','none','FaceAlpha',0.92);
    hMed = plot(ax,t,q50,'--', ...
        'Color',darken(dirCols(j,:),0.25),'LineWidth',1.5);
    hBase = plot(ax,base_flux_t,b.spillover(:,j),'-', ...
        'Color',dirCols(j,:),'LineWidth',2.8);

    xlim(ax,[t(1) t(end)]);
    xticks(ax,[0 60 120 180 236]);
    nice_axes(ax,AXCOL,FS);
    ylabel(ax,'Spillover / day','FontSize',FS,'Color',TXT);

    text(ax,0.015,0.90,dirLabs{j}, ...
        'Units','normalized','HorizontalAlignment','left', ...
        'VerticalAlignment','top','FontSize',FS_LG, ...
        'FontWeight','bold','Color',dirCols(j,:));

    if j==1
        set(ax,'XTickLabel',[]);
        legD = [reps(1) h95 h50 hMed hBase];
    else
        xlabel(ax,'Day','FontSize',FS,'Color',TXT);
    end
end

lgD = legend(firstD,legD, ...
    {'Representative solutions','5–95% band','25–75% band', ...
     'Ensemble median','Baseline'}, ...
    'NumColumns',3,'Box','off','FontSize',FS_SM);
lgD.Position = [0.245 0.875 0.53 0.055];

save_panel(figD,output_dir,'Fig3_d');

% -------------------------------------------------------------------------
% return output paths
% -------------------------------------------------------------------------
out.output_dir = output_dir;
out.files = {
    fullfile(output_dir,'Fig3_a.fig')
    fullfile(output_dir,'Fig3_a.png')
    fullfile(output_dir,'Fig3_b.fig')
    fullfile(output_dir,'Fig3_b.png')
    fullfile(output_dir,'Fig3_c.fig')
    fullfile(output_dir,'Fig3_c.png')
    fullfile(output_dir,'Fig3_d.fig')
    fullfile(output_dir,'Fig3_d.png')
    };

end


% =========================================================================
% helpers
% =========================================================================
function q = quantile_curve(vals,p)
[~,T] = size(vals);
q = nan(1,T);
for it=1:T
    q(it)=quantile_scalar(vals(:,it),p);
end
end


function q = quantile_scalar(x,p)
x=x(:);
x=x(isfinite(x));
if isempty(x), q=NaN; return; end
x=sort(x);
n=numel(x);
if n==1, q=x; return; end
pos=1+(n-1)*p;
lo=floor(pos); hi=ceil(pos);
if lo==hi
    q=x(lo);
else
    a=pos-lo;
    q=(1-a)*x(lo)+a*x(hi);
end
end


function idx = representative_indices(score,nRep)
score=score(:);
[~,ord]=sort(score,'ascend');
qq=round(linspace(1,numel(ord),nRep));
idx=unique(ord(qq),'stable');
if numel(idx)<nRep
    extra=setdiff(ord,idx,'stable');
    idx=[idx;extra(1:nRep-numel(idx))];
end
end


function draw_raincloud_box(ax,y,color,yMax,seed)
y=y(:);
y=y(isfinite(y));

% left half-violin
yg=linspace(0,yMax,300);
dens=kde1d(y,yg);
if max(dens)>0, dens=dens/max(dens); end
halfW=0.30*dens;
x0=-0.04;

fill(ax,[x0*ones(size(yg)) fliplr(x0-halfW)], ...
        [yg fliplr(yg)],color, ...
        'FaceAlpha',0.45,'EdgeColor','none');

plot(ax,x0-halfW,yg,'Color',darken(color,0.18),'LineWidth',1.4);

% jittered points
rng(seed);
xs=0.06+0.24*rand(size(y));
hs=scatter(ax,xs,y,30, ...
    'MarkerFaceColor',color, ...
    'MarkerEdgeColor',darken(color,0.30), ...
    'LineWidth',0.45);
try
    hs.MarkerFaceAlpha=0.64;
    hs.MarkerEdgeAlpha=0.30;
catch
end

% summary
q05=quantile_scalar(y,0.05);
q25=quantile_scalar(y,0.25);
q50=quantile_scalar(y,0.50);
q75=quantile_scalar(y,0.75);
q95=quantile_scalar(y,0.95);

xc=0.56;
plot(ax,[xc xc],[q05 q95],'-','Color',darken(color,0.35),'LineWidth',1.3);
plot(ax,[xc-0.05 xc+0.05],[q05 q05],'-','Color',darken(color,0.35),'LineWidth',1.3);
plot(ax,[xc-0.05 xc+0.05],[q95 q95],'-','Color',darken(color,0.35),'LineWidth',1.3);

rectangle(ax,'Position',[xc-0.08,q25,0.16,max(q75-q25,eps)], ...
    'FaceColor',lighten(color,0.35), ...
    'EdgeColor',darken(color,0.35),'LineWidth',1.2);

plot(ax,[xc-0.08 xc+0.08],[q50 q50],'-', ...
    'Color',darken(color,0.50),'LineWidth',1.9);

plot(ax,xc,q50,'o','MarkerFaceColor',darken(color,0.55), ...
    'MarkerEdgeColor','w','LineWidth',0.8,'MarkerSize',6);

text(ax,0.98,0.95,sprintf('med = %.3f',q50), ...
    'Units','normalized','HorizontalAlignment','right', ...
    'VerticalAlignment','top','FontSize',11, ...
    'Color',darken(color,0.45));
end


function draw_top_density_B(ax,x,xLim,color)
x = x(isfinite(x));
if numel(x) < 3, return; end
xx = linspace(xLim(1),xLim(2),260);
dd = kde1d(x,xx);
if max(dd) > 0
    dd = dd ./ max(dd);
end

fill(ax,[xx fliplr(xx)],[dd zeros(size(dd))],color, ...
    'FaceAlpha',0.28, ...
    'EdgeColor','none');
plot(ax,xx,dd,'Color',darken(color,0.28),'LineWidth',1.25);

xlim(ax,xLim); ylim(ax,[0 1.02]);
set(ax,'Color','none', ...
    'Box','off', ...
    'FontName','Arial', ...
    'LineWidth',1.0, ...
    'XColor','k','YColor','k', ...
    'XTick',[],'YTick',[], ...
    'TickDir','out', ...
    'Layer','top');
axis(ax,'off');
end


function draw_right_density_B(ax,y,yLim,color)
y = y(isfinite(y));
if numel(y) < 3, return; end
yy = linspace(yLim(1),yLim(2),260);
dd = kde1d(y,yy);
if max(dd) > 0
    dd = dd ./ max(dd) * 0.92;
end

fill(ax,[zeros(size(dd)) fliplr(dd)],[yy fliplr(yy)],color, ...
    'FaceAlpha',0.28, ...
    'EdgeColor','none');
plot(ax,dd,yy,'Color',darken(color,0.28),'LineWidth',1.25);

xlim(ax,[0 1.0]); ylim(ax,yLim);
set(ax,'Color','none', ...
    'Box','off', ...
    'FontName','Arial', ...
    'LineWidth',1.0, ...
    'XColor','k','YColor','k', ...
    'XTick',[],'YTick',[], ...
    'TickDir','out', ...
    'Layer','top');
axis(ax,'off');
end


function dens=kde1d(x,grid)
x=x(:); x=x(isfinite(x));
n=numel(x);
if n<2, dens=zeros(size(grid)); return; end

sx=std(x);
iq=iqr_basic(x);
sigma=min(sx,iq/1.349);
if ~isfinite(sigma)||sigma<=0
    sigma=max(sx,1e-3);
end
h=max(0.9*sigma*n^(-1/5),1e-4);

dens=zeros(size(grid));
for i=1:n
    z=(grid-x(i))/h;
    dens=dens+exp(-0.5*z.^2);
end
dens=dens/(n*h*sqrt(2*pi));
end


function v=iqr_basic(x)
x=sort(x(:));
v=quantile_scalar(x,0.75)-quantile_scalar(x,0.25);
end


function save_panel(fig,output_dir,name)
savefig(fig,fullfile(output_dir,[name '.fig']));
exportgraphics(fig,fullfile(output_dir,[name '.png']),'Resolution',300);
end


function c=lighten(c0,amt)
c=c0+(1-c0)*amt;
c=min(max(c,0),1);
end


function c=darken(c0,amt)
c=c0.*(1-amt);
c=min(max(c,0),1);
end


function nice_axes(ax,axisColor,fontSize)
set(ax, ...
    'FontName','Arial', ...
    'FontSize',fontSize, ...
    'LineWidth',0.9, ...
    'TickDir','out', ...
    'TickLength',[0.012 0.012], ...
    'Box','off', ...
    'Layer','top', ...
    'XColor',axisColor, ...
    'YColor',axisColor);
grid(ax,'off');
end


function cmap=make_distance_cmap(n)
anchors=[
    0.11 0.24 0.35
    0.16 0.51 0.55
    0.43 0.74 0.73
    0.74 0.84 0.71
    0.93 0.89 0.63
];
x0=linspace(0,1,size(anchors,1));
xq=linspace(0,1,n);
cmap=zeros(n,3);
for j=1:3
    cmap(:,j)=interp1(x0,anchors(:,j),xq,'pchip');
end
cmap=max(0,min(1,cmap));
end
