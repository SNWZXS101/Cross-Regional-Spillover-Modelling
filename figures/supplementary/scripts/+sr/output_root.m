function p=output_root()
p=getenv('UDE_SUPPLEMENT_OUTPUT_ROOT');
if isempty(p), p=fullfile(sr.root(),'rendered'); end
end
