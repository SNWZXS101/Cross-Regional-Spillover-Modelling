function timeaxis(ax,label)
if nargin<2,label=true;end
xlim(ax,[0 236]);xticks(ax,[0 60 120 180 236]);
if label,xlabel(ax,'Day');else,ax.XTickLabel=[];end
end
