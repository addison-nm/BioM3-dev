"""Console-script shims for the multidomain subpackage."""

import sys


def run_multidomain_finetuning():
    from biom3.Stage3.multidomain.run_multidomain_finetuning import (
        main, parse_arguments,
    )
    args = parse_arguments(sys.argv[1:])
    sys.exit(main(args))


if __name__ == "__main__":
    run_multidomain_finetuning()
