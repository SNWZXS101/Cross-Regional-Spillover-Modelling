function d = load_data()
% Verify the portable .mat snapshot before every independently runnable figure.
p=fullfile(mr.root(),'data','main_results.mat');
manifest=jsondecode(fileread(fullfile(mr.root(),'data','input_manifest.json')));
if manifest.manuscript_contract.manuscript_main_n~=100 || ~strcmp(manifest.manuscript_contract.manuscript_main_source,'analysis_runs/sensitivity_100')
    error('MainResults:ManuscriptRole','The current manuscript main ensemble must be frozen N=100.');
end
fid=fopen(p,'rb'); assert(fid>=0,'Missing MATLAB input snapshot.');
c=onCleanup(@()fclose(fid)); bytes=fread(fid,Inf,'*uint8');
md=java.security.MessageDigest.getInstance('SHA-256'); md.update(bytes);
actual=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
assert(strcmp(actual,manifest.mat_sha256),'Input SHA256 mismatch.');
d=load(p); d.strategies=cellstr(d.strategies); d.measures=cellstr(d.measures);
end
