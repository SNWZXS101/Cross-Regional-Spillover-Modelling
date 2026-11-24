function s=style()
% Same Arial and restrained palette as MainResults/matlab/scripts/+mr/style.m.
s.GD=hex('194F78');s.GX=hex('087C78');s.benefit=hex('258E98');
s.harm=hex('C69454');s.neutral=hex('A4ACB0');s.ink=hex('243C4A');
s.font='Arial';s.fs=9;s.dpi=600;
s.fraction=interp1([0 .33 .67 1],[hex('F5F3EB');hex('99CDD1');hex('287F95');hex('173E59')],linspace(0,1,256));
set(groot,'defaultAxesFontName',s.font,'defaultTextFontName',s.font,...
 'defaultAxesFontSize',s.fs,'defaultTextFontSize',s.fs,'defaultAxesLineWidth',.6,...
 'defaultAxesTickDir','out','defaultAxesBox','off','defaultFigureColor','w');
end
function c=hex(x)
c=sscanf(x,'%2x%2x%2x',[1 3])/255;
end
