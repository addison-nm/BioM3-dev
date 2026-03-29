import sys


def main():
    print("Hello world from biom3.data_prep.__main__")


def run_compile_hdf5():
    from biom3.data_prep.compile_stage2_data_to_hdf5 import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    main()
