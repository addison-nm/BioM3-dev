import sys

def main():
    print("Hello world from Stage1_source.__main__")

def run_PenCL_inference():
    from Stage1_source.run_PenCL_inference import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    main()
