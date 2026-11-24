function f=make_tableS2()
d=sr.load_data();headers={'A / B',sprintf('Reported\nN = 30'),sprintf('Reported\nN = 50'),sprintf('Reported\nN = 100'),sprintf('Diagnostic\nN = 30'),sprintf('Diagnostic\nN = 50'),sprintf('Diagnostic\nN = 100'),sprintf('N = 100 lower burden\nReported'),sprintf('N = 100 lower burden\nDiagnostic')};
f=sr.tablefigure(headers,d.table2,[.080 .085 .085 .085 .085 .085 .085 .205 .205],12.5,8);
annotation(f,'textbox',[.042 .01 .93 .04],'String','Count/N columns give A < B. Last columns give each lower-burden strategy and its member count; both names indicate reversals. Exact reverse counts, ties and fractions are in CSV.','EdgeColor','none','FontSize',8);
sr.save(f,'tables','TableS2');copyfile(fullfile(sr.root(),'data','TableS2_exact.csv'),fullfile(sr.output_root(),'tables','TableS2.csv'));
end
