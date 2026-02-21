import sys

def main():
    print("Hello world from biom3.Stage2.__main__")

def run_Facilitator_sample():
    from biom3.Stage2.run_Facilitator_sample import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)

if __name__ == "__main__":
    main()
