function signedaxis(ax,which,limits,threshold,linscale)
% Original-value tick labels on the explicitly transformed display coordinates.
if nargin<5,linscale=.6;end
maxexp=ceil(log10(max(abs(limits))));
positive=10.^(floor(log10(threshold)):maxexp);
ticks=[-fliplr(positive),0,positive];ticks=ticks(ticks>=limits(1)&ticks<=limits(2));
set(ax,[upper(which) 'Lim'],mr.symlog(limits,threshold,linscale),...
    [upper(which) 'Tick'],mr.symlog(ticks,threshold,linscale),...
    [upper(which) 'TickLabel'],arrayfun(@(x)sprintf('%g',x),ticks,'UniformOutput',false));
info=ax.UserData;info.([which '_display_transform'])=struct('name','symlog','threshold',threshold,'linscale',linscale);
ax.UserData=info;
end
