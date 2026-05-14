import sys

def main():
    print("Hello world from biom3.decode_llama.__main__")

def run_ZpDecoder_training():
    from biom3.decode_llama.run_ZpDecoder_training import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)

def run_caption():
    from biom3.decode_llama.query import _cli
    _cli(sys.argv[1:])

if __name__ == "__main__":
    main()
