function save(f,name,kind)
if nargin<3,kind='figures';end
outputRoot=getenv('UDE_FIGURE_OUTPUT_ROOT');
if isempty(outputRoot),outputRoot=mr.root();end
folder=fullfile(outputRoot,kind);if ~isfolder(folder),mkdir(folder);end
axs=findall(f,'Type','axes');
for i=1:numel(axs),assert(isempty(axs(i).Title.String),'Keep titles in external captions.');end
drawnow;savefig(f,fullfile(folder,[name '.fig']),'compact');
s=mr.style();exportgraphics(f,fullfile(folder,[name '.png']),'Resolution',s.dpi,'BackgroundColor','white');
fprintf('%s: native editable FIG + %d dpi PNG saved.\n',name,s.dpi);
end
