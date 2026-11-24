function f=plot_figS1()
% Saved manifest fits and full frozen distance matrices. No solver calls.
s=sr.style();d=sr.load_data();f=figure('Units','inches','Position',[1 1 10 6.8],'Color','w');
regions={'Guangdong','Guangxi'};cols=[s.GD;s.GX];
for r=1:2
 ax=axes(f,'Position',[.09+(r-1)*.48 .57 .40 .35]);hold(ax,'on');
 for ni=1:3
  n=d.sizes(ni);q=d.(sprintf('r2_%d',n));
  for solver=1:2
   y=q(:,r+2*(solver-1));x=ni+(solver-1.5)*.25;
   color=cols(r,:);if solver==2,color=.48+.52*color;end
   jitter=.055*sin((1:n)'*2.3999632297);
   marker='o';if solver==2,marker='^';end
   scatter(ax,x+jitter,y,10,color,marker,'filled','MarkerFaceAlpha',.45,'HandleVisibility','off');
   v=sort(y);quart=interp1(linspace(0,1,n),v,[.25 .5 .75]);
   rectangle(ax,'Position',[x-.074 quart(1) .148 max(quart(3)-quart(1),eps)],'EdgeColor',cols(r,:),'LineWidth',.8);
   plot(ax,[x-.074 x+.074],[quart(2) quart(2)],'Color',cols(r,:),'LineWidth',1.3,'HandleVisibility','off');
  end
 end
 h1=plot(ax,nan,nan,'o','Color',cols(r,:),'MarkerFaceColor',cols(r,:),'MarkerSize',4);
 h2=plot(ax,nan,nan,'^','Color',.48+.52*cols(r,:),'MarkerFaceColor',.48+.52*cols(r,:),'MarkerSize',4);
 yline(ax,.98,'--','Color',s.harm,'LineWidth',1,'HandleVisibility','off');
 text(ax,.04,.04,'Acceptance: R^2 > 0.98','Units','normalized','Color',s.harm,'FontSize',8);
 text(ax,.04,.93,regions{r},'Units','normalized','Color',cols(r,:),'FontWeight','bold');
 set(ax,'XLim',[.5 3.5],'XTick',1:3,'XTickLabel',{'30','50','100'},'YLim',[.9785 1.0005],'YTick',.98:.005:1);
 xlabel(ax,'Ensemble size, N');ylabel(ax,'R^2 from independent ODE integration');
 legend(ax,[h1 h2],{'RK4','DOP853'},'Location','northeast','Box','off','FontSize',8);
 sr.panel(ax,char('a'+r-1));
 % A linear-offset inset resolves tiny member differences without hiding the
 % scientific threshold on the main axes. It uses the same saved R2 values.
 zoom=axes(f,'Position',[.245+(r-1)*.48 .655 .225 .12]);hold(zoom,'on');
 offset=.99338;if r==2,offset=.99025;end
 for ni=1:3
  n=d.sizes(ni);q=d.(sprintf('r2_%d',n));
  for solver=1:2
   y=(q(:,r+2*(solver-1))-offset)*1e6;x=ni+(solver-1.5)*.25;
   color=cols(r,:);marker='o';if solver==2,color=.48+.52*color;marker='^';end
   scatter(zoom,x+.055*sin((1:n)'*2.3999632297),y,6,color,marker,'filled','MarkerFaceAlpha',.5);
   quart=interp1(linspace(0,1,n),sort(y),[.25 .5 .75]);
   rectangle(zoom,'Position',[x-.07 quart(1) .14 quart(3)-quart(1)],'EdgeColor',cols(r,:),'LineWidth',.65);
   plot(zoom,[x-.07 x+.07],[quart(2) quart(2)],'Color',cols(r,:),'LineWidth',1);
  end
 end
 set(zoom,'XLim',[.5 3.5],'XTick',1:3,'XTickLabel',{'30','50','100'},'FontSize',6.5,'Color','w');
 ylabel(zoom,sprintf('(R^2 - %.5f) x 10^6',offset),'FontSize',6.5);
end
edges=linspace(0,.40,31);
for ni=1:3
 ax=axes(f,'Position',[.08+(ni-1)*.31 .105 .255 .31]);hold(ax,'on');n=d.sizes(ni);v=d.(sprintf('distances_%d',n));
 histogram(ax,v,edges,'FaceColor',s.benefit,'FaceAlpha',.65,'EdgeColor','white','LineWidth',.3);
 xlim(ax,[0 .40]);xlabel(ax,'Functional distance, D');ylabel(ax,'Solution pairs');
 text(ax,.97,.96,sprintf('N = %d; %d pairs\nmin %.4f\nmedian %.4f\nmax %.4f',n,numel(v),min(v),median(v),max(v)),...
  'Units','normalized','HorizontalAlignment','right','VerticalAlignment','top','FontSize',8,'BackgroundColor','w','Margin',1);
 sr.panel(ax,char('c'+ni-1));
end
sr.save(f,'figures','FigS1');
end
