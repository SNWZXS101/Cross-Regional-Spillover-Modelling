function y = symlog(x,threshold,linscale)
% Same base-10 symmetric-log display mapping as the Python figures.
% Scientific data are unchanged. Use invsymlog to recover source units.
if nargin<3,linscale=.6;end
a=linscale/(1-1/10); y=x*a; outside=abs(x)>threshold;
y(outside)=sign(x(outside))*threshold.*(a+log10(abs(x(outside))/threshold));
end
