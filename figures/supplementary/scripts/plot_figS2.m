function f=plot_figS2()
% Descriptive 5th--95th percentile bands on all saved rate-grid points.
s=sr.style();d=sr.load_data();f=figure('Units','inches','Position',[1 1 8.5 9.5],'Color','w');
names={'c_1','c_2','q_1','q_2','m_1 (day^{-1})','m_2 (day^{-1})','d_1 (day^{-1})','d_2 (day^{-1})'};
% m is dimensionless in the model; retain original scientific units.
names{5}='m_1';names{6}='m_2';t=d.time(:);mix=[.75 .48 .08];handles=gobjects(1,4);regions={'Guangdong','Guangxi'};
for p=1:8
 row=ceil(p/2);col=mod(p-1,2);ax=axes(f,'Position',[.10+col*.48 .76-(row-1)*.222 .39 .17]);hold(ax,'on');
 color=s.GD;if col==1,color=s.GX;end
 for ni=1:3
  lo=d.bands(1,:,p,ni);hi=d.bands(2,:,p,ni);c=mix(ni)+(1-mix(ni))*color;
  h=fill(ax,[t;flipud(t)],[lo(:);flipud(hi(:))],c,'EdgeColor','none','FaceAlpha',.38);
  plot(ax,t,lo,'Color',c,'LineWidth',.45,'HandleVisibility','off');plot(ax,t,hi,'Color',c,'LineWidth',.45,'HandleVisibility','off');
  if p==1,handles(ni)=h;end
 end
 h=plot(ax,d.baseline_time,d.baseline_rates(:,p),'Color',color,'LineWidth',1.4);if p==1,handles(4)=h;end
 xlim(ax,[0 236]);ylim(ax,d.bounds(p,:));xticks(ax,[0 59 118 177 236]);ylabel(ax,names{p});
 if row==4,xlabel(ax,'Day');else,set(ax,'XTickLabel',[]);end
 sr.panel(ax,char('a'+p-1));
 if row==1,text(ax,.98,.91,regions{col+1},'Units','normalized','HorizontalAlignment','right','Color',color,'FontWeight','bold');end
end
lg=legend(handles,{'N = 30','N = 50','N = 100','Baseline'},'Orientation','horizontal','Box','off','FontSize',9);
set(lg,'Units','normalized','Position',[.20 .953 .62 .027]);
sr.save(f,'figures','FigS2');
end
