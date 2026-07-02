import sys


def run_cluster_split():
    from biom3.split.run_split import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)


def run_stratified_cluster_split():
    from biom3.split.run_stratified_split import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    run_cluster_split()
