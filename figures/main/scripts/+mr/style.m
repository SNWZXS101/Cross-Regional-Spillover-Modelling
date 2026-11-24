function s = style()
% Edit this file to change the palette and typography for all seven objects.
s.GD=rgb('194F78'); s.GX=rgb('087C78'); s.cyan=rgb('46A4BB');
s.green=rgb('56A58C'); s.navy=rgb('183D54'); s.benefit=rgb('258E98');
s.harm=rgb('C69454'); s.gold=rgb('DFC994'); s.neutral=rgb('A4ACB0');
s.ink=rgb('243C4A'); s.regions=[s.GD;s.GX]; s.font='Arial'; s.fontsize=7;
s.width=180/25.4; s.dpi=450; s.linewidth=1;
s.ensemble=interp1([0 1],[rgb('89BBD0');rgb('A4CABC')],linspace(0,1,100));
s.distance=interp1(linspace(0,1,4),[rgb('173A54');rgb('23909C');rgb('93C9C9');rgb('F1E4B3')],linspace(0,1,256));
s.fraction=interp1(linspace(0,1,4),[rgb('F5F3EB');rgb('99CDD1');rgb('287F95');rgb('173E59')],linspace(0,1,256));
s.signed=interp1([0 .5 1],[s.harm;rgb('F9F7EE');s.benefit],linspace(0,1,256));
end
function c=rgb(hex)
c=sscanf(hex,'%2x%2x%2x',[1 3])/255;
end
