function run_all_main_results(output_root)
% Render from saved inputs only. Output goes to a separate review directory.
% Fig. 3--5 panel figures are saved individually; their manuscript composites
% remain curated separately in manuscript/figures.
root=fileparts(mfilename('fullpath'));addpath(fullfile(root,'scripts'));
if nargin<1 || isempty(output_root),output_root=fullfile(root,'rendered');end
if ~isfolder(output_root),mkdir(output_root);end
old=getenv('UDE_FIGURE_OUTPUT_ROOT');cleanup=onCleanup(@()setenv('UDE_FIGURE_OUTPUT_ROOT',old));
setenv('UDE_FIGURE_OUTPUT_ROOT',output_root);
f=plot_fig2();close(f);
plot_fig3(fullfile(output_root,'Fig3_panels'));close all;
run_panel_script(fullfile(root,'scripts','plot_fig4.m'));
run_panel_script(fullfile(root,'scripts','plot_fig5_final.m'));
f=make_table1();close(f);f=make_table2();close(f);
fprintf('Generated saved-data display outputs in %s.\n',output_root);
end
function run_panel_script(path)
run(path);
end
