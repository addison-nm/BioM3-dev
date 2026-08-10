import sys


def run_fit_manifold():
    from biom3.geometry.run_fit_manifold import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)


def run_score_manifold():
    from biom3.geometry.run_score_manifold import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    run_fit_manifold()
