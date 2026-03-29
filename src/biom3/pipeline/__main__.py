import sys


def main():
    print("Hello world from biom3.pipeline.__main__")


def run_embedding_pipeline():
    from biom3.pipeline.embedding_pipeline import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    main()
