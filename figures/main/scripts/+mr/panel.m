function h = panel(ax,letter,x)
if nargin<3,x=-.15;end
h=text(ax,x,1.055,letter,'Units','normalized','FontSize',9,'FontWeight','bold',...
    'VerticalAlignment','bottom','Clipping','off','Color',mr.style().ink);
end
