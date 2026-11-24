function q = quantiles(x,p)
% NumPy-compatible linear empirical quantiles; no Statistics Toolbox needed.
x=sort(x(:));h=1+(numel(x)-1)*p;lo=floor(h);hi=ceil(h);
q=reshape(x(lo),size(p)).*(hi-h)+reshape(x(hi),size(p)).*(h-lo);
q(lo==hi)=x(lo(lo==hi));
end
