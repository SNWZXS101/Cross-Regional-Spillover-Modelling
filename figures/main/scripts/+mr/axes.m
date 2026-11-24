function ax = axes(f,position)
s=mr.style();
ax=axes('Parent',f,'Units','normalized','Position',position,'Box','off',...
    'FontName',s.font,'FontSize',s.fontsize,'LineWidth',.55,'TickDir','out',...
    'XColor',s.ink,'YColor',s.ink,'Color','w','NextPlot','add');
ax.Toolbar.Visible='off';
end
