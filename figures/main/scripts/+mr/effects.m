function values = effects(d,n,measure)
% Columns: G_HH-LL, A_total_HH, A_GD_HN, A_GX_NH. Member order is U01...UN.
v=d.(sprintf('run%d',n)).values;
values=[v(:,6,measure,5)-v(:,9,measure,5),v(:,6,measure,5),v(:,2,measure,3),v(:,4,measure,4)];
end
