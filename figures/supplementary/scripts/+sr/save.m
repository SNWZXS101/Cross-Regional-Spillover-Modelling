function save(f,folder,name)
s=sr.style();p=fullfile(sr.output_root(),folder);if ~isfolder(p),mkdir(p);end
set(f,'PaperPositionMode','auto');drawnow;
savefig(f,fullfile(p,[name '.fig']));
exportgraphics(f,fullfile(p,[name '.png']),'Resolution',s.dpi,'BackgroundColor','white');
fprintf('Saved %s: editable FIG and 600-dpi PNG\n',name);
end
