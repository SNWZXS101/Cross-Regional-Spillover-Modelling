function d=load_data()
p=fullfile(sr.root(),'data','supplementary_data.mat');
m=jsondecode(fileread(fullfile(sr.root(),'data','input_sha256.json')));
fid=fopen(p,'rb');if fid<0,error('Missing supplemental input MAT');end
cleanup=onCleanup(@()fclose(fid));bytes=fread(fid,Inf,'*uint8');
md=java.security.MessageDigest.getInstance('SHA-256');md.update(bytes);
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
if ~strcmp(h,m.supplementary_data_mat),error('Supplemental MAT input hash mismatch');end
d=load(p);
d.sizes=double(d.sizes);d.counts=double(d.counts);d.ties=double(d.ties);
end
