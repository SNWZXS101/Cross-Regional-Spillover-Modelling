function f = tablefigure(name,headers,rows,widths,height,fontsize)
% Native editable text and rules, not a screenshot or embedded HTML image.
f=mr.newfigure(name,height,11.7);s=mr.style();
ax=mr.axes(f,[.03 .025 .94 .95]);axis(ax,[0 1 0 1]);axis(ax,'off');
edges=[0 cumsum(widths/sum(widths))];n=size(rows,1);rowheight=1/(n+1.15);
for i=1:n+1
    top=1-(i-1)*rowheight;
    if i>1 && mod(i,2)==1,patch(ax,[0 1 1 0],[top top top-rowheight top-rowheight],[.95 .97 .97],'EdgeColor','none');end
    for j=1:numel(headers)
        if i==1,txt=headers{j};weight='bold';else,txt=rows{i-1,j};weight='normal';end
        % Only formatting: wrap source strings to fit the editable table cells.
        maxchars=max(8,floor((edges(j+1)-edges(j))*11.0*72/(fontsize*.50)));
        txt=char(string(txt));interpreter='none';
        if startsWith(txt,'tex:'),txt=txt(5:end);interpreter='tex';else,txt=wraptext(txt,maxchars);end
        text(ax,edges(j)+.008,top-rowheight*.16,txt,'FontName',s.font,'FontSize',fontsize,...
            'FontWeight',weight,'Interpreter',interpreter,'VerticalAlignment','top','Color',s.ink);
    end
end
for y=[1 1-rowheight 1-(n+1)*rowheight]
    plot(ax,[0 1],[y y],'Color',s.navy,'LineWidth',.8);
end
f.UserData.Headers=headers;f.UserData.Cells=rows;
end
function txt=wraptext(txt,n)
words=regexp(txt,'\s+','split');lines={''};
for k=1:numel(words)
    if numel(lines{end})+numel(words{k})+1>n && ~isempty(lines{end}),lines{end+1}=words{k}; %#ok<AGROW>
    elseif isempty(lines{end}),lines{end}=words{k};else,lines{end}=[lines{end} ' ' words{k}];end
end
txt=strjoin(lines,newline);
end
