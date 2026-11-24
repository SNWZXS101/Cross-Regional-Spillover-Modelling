function [xy,stats] = classical_mds(distance)
n=size(distance,1);H=eye(n)-ones(n)/n;
B=-.5*H*(distance.^2)*H;[vec,val]=eig((B+B')/2,'vector');
[val,order]=sort(val,'descend');vec=vec(:,order);xy=vec(:,1:2).*sqrt(max(val(1:2)',0));
for j=1:2,[~,k]=max(abs(xy(:,j)));if xy(k,j)<0,xy(:,j)=-xy(:,j);end,end
fitted=hypot(xy(:,1)-xy(:,1)',xy(:,2)-xy(:,2)');mask=triu(true(n),1);
stats.positive_inertia_2d=sum(max(val(1:2),0))/sum(val(val>0));
stats.negative_eigenvalue_fraction=sum(abs(val(val<0)))/sum(abs(val));
stats.normalized_stress=sqrt(sum((distance(mask)-fitted(mask)).^2)/sum(distance(mask).^2));
end
