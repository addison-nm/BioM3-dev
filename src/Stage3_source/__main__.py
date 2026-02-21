import sys

def main():
    print("Hello world from Stage3_source.__main__")

def run_ProteoScribe_sample():
    from Stage3_source.run_ProteoScribe_sample import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    main()
