import sys


def run_benchmark_stage3_generation():
    from biom3.benchmarks.Stage3.generation import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)


def run_plot_benchmark():
    from biom3.viz.benchmark import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)
