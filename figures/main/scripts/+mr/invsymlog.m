function x = invsymlog(y,threshold,linscale)
if nargin<3,linscale=.6;end
a=linscale/(1-1/10);x=y/a;outside=abs(y)>threshold*a;
x(outside)=sign(y(outside))*threshold.*10.^(abs(y(outside))/threshold-a);
end
