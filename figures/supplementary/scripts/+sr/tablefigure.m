function f=tablefigure(headers,rows,widths,height,fs)
s=sr.style();f=figure('Units','inches','Position',[1 1 12 height],'Color','w');
ax=axes(f,'Position',[.035 .06 .93 .88]);axis(ax,[0 1 0 1]);axis(ax,'off');hold(ax,'on');
nr=size(rows,1);head=1;if nr>20,head=2.3;end;dy=.92/(nr+head+.5);x=[0 cumsum(widths)];
for j=1:numel(headers)
 text(ax,x(j)+.009,.98,headers{j},'VerticalAlignment','top','FontSize',fs,'FontWeight','bold','Interpreter','none','Color',s.ink);
end
plot(ax,[0 1],[.98-head*dy .98-head*dy],'Color',s.ink,'LineWidth',.9);
for i=1:nr
 y=.98-dy*(head+i-.85);
 if mod(i,2)==0,patch(ax,[0 1 1 0],[y+.08*dy y+.08*dy y-.92*dy y-.92*dy],[.955 .975 .975],'EdgeColor','none');end
 for j=1:numel(headers)
  text(ax,x(j)+.009,y,rows{i,j},'VerticalAlignment','top','FontSize',fs,'Interpreter','none','Color',s.ink);
 end
end
plot(ax,[0 1],[.98-dy*(nr+head+.1) .98-dy*(nr+head+.1)],'Color',s.ink,'LineWidth',.7);
end
