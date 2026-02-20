import sys

def main():
    print("Hello world from Stage2_source.__main__")

def run_Facilitator_sample():
    from Stage2_source.run_Facilitator_sample import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    main()
