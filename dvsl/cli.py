"""Installed command entry points; imports are deferred to the selected workflow."""
import argparse
import sys

def match_df_psm():
    from .two_stage import run_cli
    return run_cli('df_psm')

def match_dtm():
    from .two_stage import run_cli
    return run_cli('dtm')

def match_ges():
    from .two_stage import run_cli
    return run_cli('ges')

def recover_alpha():
    from .alpha_recovery import cli
    return cli()

def generate():
    from .velocity_diffusion import cli
    return cli()

def basis():
    from .mercer_basis import cli
    return cli()

def ot():
    from .ot_preprocessing import cli
    return cli()

def preprocess():
    from .surface_preprocessing import main
    return main()

def preprocess_corresponding():
    from .surface_preprocessing import main
    return main(default_mode='corresponding')

def preprocess_noncorresponding():
    from .surface_preprocessing import main
    return main(default_mode='noncorresponding')

COMMANDS = {
    'match-df-psm': match_df_psm, 'match-dtm': match_dtm, 'match-ges': match_ges,
    'recover-alpha': recover_alpha, 'generate': generate, 'basis': basis, 'ot': ot,
    'preprocess': preprocess, 'preprocess-corresponding': preprocess_corresponding,
    'preprocess-noncorresponding': preprocess_noncorresponding,
}

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ('-h','--help'):
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command',choices=tuple(COMMANDS))
        parser.print_help()
        return 0
    command=sys.argv.pop(1)
    if command not in COMMANDS:
        print('Unknown command: '+command, file=sys.stderr)
        return 2
    return COMMANDS[command]()

if __name__ == '__main__':
    raise SystemExit(main())
