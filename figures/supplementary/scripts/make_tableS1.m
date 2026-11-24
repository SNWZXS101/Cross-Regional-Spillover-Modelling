function f=make_tableS1()
d=sr.load_data();rows=d.table1;rows{3,1}='RK4 R2, Guangdong';rows{4,1}='RK4 R2, Guangxi';rows{5,1}='DOP853 R2, Guangdong';rows{6,1}='DOP853 R2, Guangxi';
f=sr.tablefigure({'Diagnostic','N = 30','N = 50','N = 100'},rows,[.28 .24 .24 .24],4.9,9);
annotation(f,'textbox',[.045 .025 .9 .04],'String','Fit quality and distance entries: minimum / median / maximum. Full precision is retained in CSV.','EdgeColor','none','FontSize',9);
sr.save(f,'tables','TableS1');copyfile(fullfile(sr.root(),'data','TableS1_exact.csv'),fullfile(sr.output_root(),'tables','TableS1.csv'));
end
