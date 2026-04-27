import sys

def main():
    print("Hello world from biom3.Stage3.__main__")

def run_ProteoScribe_sample():
    from biom3.Stage3.run_ProteoScribe_sample import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)

def run_stage3_training():
    from biom3.Stage3.run_PL_training import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    main()
