function counts = robustness(d,n,measure)
% Exact original strict comparison and 0.01-case classification tolerance.
r=d.(sprintf('run%d',n));v=r.values; c=r.classification;t=.01;
counts=[sum(v(:,6,measure,5)>v(:,9,measure,5));sum(v(:,9,measure,5)<-t);...
    sum(v(:,9,measure,3)<-t & v(:,9,measure,4)<-t);sum(strcmp(c(:,6,measure),'win-win'));...
    sum(v(:,6,measure,5)>t)];
for k=[2 4 3 5],counts(end+1,1)=sum(strcmp(c(:,k,measure),'self-harm / neighbor-benefit'));end
for k=[7 8],counts(end+1,1)=sum(strcmp(c(:,k,measure),'win-win'));end
end
