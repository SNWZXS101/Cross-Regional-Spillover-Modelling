function f = newfigure(name,height,width)
s=mr.style(); if nargin<3,width=s.width;end
f=figure('Name',name,'NumberTitle','off','Color','w','Units','inches',...
    'Position',[.7 .7 width height],'PaperPositionMode','auto');
f.UserData=struct('Source','Saved MainResults numerical inputs','ODECalls',0,...
    'Note','Editable native MATLAB graphics. Captions are external.');
end
