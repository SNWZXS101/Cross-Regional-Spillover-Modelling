function f=plot_figS3()
% Strict row<column burden comparisons from saved scenarios; ties not wins.
s=sr.style();d=sr.load_data();f=figure('Units','inches','Position',[1 1 12 8.4],'Color','w');
labels=cellstr(string(d.strategies));last=[];
for mi=1:2
 for ni=1:3
  ax=axes(f,'Position',[.085+(ni-1)*.285 .565-(mi-1)*.445 .240 .335]);n=d.sizes(ni);c=d.counts(:,:,mi,ni);
  imagesc(ax,c/n,[0 1]);colormap(ax,s.fraction);axis(ax,'image');hold(ax,'on');last=ax;
  set(ax,'XTick',1:9,'YTick',1:9,'XTickLabel',labels,'YTickLabel',labels,'TickLength',[0 0],'FontSize',8,'Box','on');
  for i=1:9
   for j=1:9
    label=sprintf('%.2f',c(i,j)/n);ink=s.ink;if c(i,j)/n>.64,ink=[1 1 1];end
    if i==j,label='--';end
    text(ax,j,i,label,'HorizontalAlignment','center','FontSize',8,'Color',ink,'Interpreter','none');
    if i~=j && (c(i,j)==n || c(j,i)==n)
     rectangle(ax,'Position',[j-.46 i-.46 .92 .92],'EdgeColor',s.harm,'LineWidth',1.2);
    end
   end
  end
  xlabel(ax,'Column strategy B');ylabel(ax,'Row strategy A');
  text(ax,.5,1.075,sprintf('N = %d   |   %d/36 unanimous pairs',n,nnz(c==n)),'Units','normalized','HorizontalAlignment','center','FontSize',9);
  sr.panel(ax,char('a'+(mi-1)*3+ni-1));
 end
end
annotation(f,'textbox',[.085 .962 .5 .025],'String','Total reported burden','EdgeColor','none','FontName','Arial','FontSize',10,'FontWeight','bold','Color',s.ink);
annotation(f,'textbox',[.085 .507 .5 .025],'String','All-infections diagnostic','EdgeColor','none','FontName','Arial','FontSize',10,'FontWeight','bold','Color',s.ink);
cb=colorbar(last,'Position',[.927 .18 .014 .60]);cb.Ticks=[0 .25 .5 .75 1];cb.Label.String='Ensemble fraction: row A < column B';cb.FontSize=8;
annotation(f,'textbox',[.18 .014 .7 .033],'String','Amber outlines: unanimous ordering (both complementary cells); diagonal excluded.','EdgeColor','none','FontSize',8,'Color',s.harm);
sr.save(f,'figures','FigS3');
end
