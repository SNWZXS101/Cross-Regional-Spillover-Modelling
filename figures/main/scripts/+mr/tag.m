function h = tag(ax,label,x,y,align)
if nargin<3,x=.97;end
if nargin<4,y=.95;end
if nargin<5,align='right';end
h=text(ax,x,y,label,'Units','normalized','FontSize',6.6,...
    'HorizontalAlignment',align,'VerticalAlignment','top','Color',mr.style().ink);
end
