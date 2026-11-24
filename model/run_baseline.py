"""Reproduce the primary UDE training or independently validate exported rates."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('validate','train'))
    p.add_argument('--output', type=Path, required=True,
                   help='Output directory for generated files (never input data)')
    p.add_argument('--data', type=Path, default=ROOT/'data/input/Data_all_sm.xlsx')
    p.add_argument('--baseline', type=Path, default=ROOT/'data/baseline')
    p.add_argument('--device', choices=('cpu','cuda','auto'), default='cpu')
    args=p.parse_args()
    output=args.output.expanduser().resolve()
    source=(ROOT/'data').resolve()
    if output==source or source in output.parents:
        p.error('Output must not be within the packaged data directory')
    baseline=args.baseline.expanduser().resolve()
    if output==baseline or baseline in output.parents:
        p.error('Output must not be within the baseline input directory')
    if args.mode=='validate':
        import independent_ode_validation as engine
        engine.RESULTS_DIR=str(args.baseline.resolve())
        engine.DATA_FILE=str(args.data.resolve())
        engine.OUTPUT_DIR=str(output)
        engine.main()
    else:
        import train_ude as engine
        engine.CONFIG.data_file=str(args.data.resolve())
        engine.CONFIG.output_root=str(output)
        engine.CONFIG.device=args.device
        engine.main()

if __name__=='__main__': main()
