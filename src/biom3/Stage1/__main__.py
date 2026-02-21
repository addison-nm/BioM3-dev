import sys

def main():
    print("Hello world from biom3.Stage1.__main__")

def run_PenCL_inference():
    from biom3.Stage1.run_PenCL_inference import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    main()
