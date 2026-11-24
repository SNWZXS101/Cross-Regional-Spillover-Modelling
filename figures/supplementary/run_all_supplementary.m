function run_all_supplementary(output_root)
% Render prepared saved artifacts; no scientific solver.
root=fileparts(mfilename('fullpath'));addpath(fullfile(root,'scripts'));
if nargin<1 || isempty(output_root), output_root=fullfile(root,'rendered'); end
if ~isfolder(output_root), mkdir(output_root); end
old=getenv('UDE_SUPPLEMENT_OUTPUT_ROOT'); cleanup=onCleanup(@()setenv('UDE_SUPPLEMENT_OUTPUT_ROOT',old));
setenv('UDE_SUPPLEMENT_OUTPUT_ROOT',output_root);
tasks={@plot_figS1,@plot_figS2,@plot_figS3,@make_tableS1,@make_tableS2};
for k=1:numel(tasks),f=tasks{k}();close(f);end
end
