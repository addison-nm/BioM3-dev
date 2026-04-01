import sys


def main():
    print("Hello world from biom3.dbio.__main__")


def run_build_dataset():
    from biom3.dbio.build_dataset import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    main(args)


def run_build_taxid_index():
    import argparse
    from biom3.dbio.taxonomy import AccessionTaxidMapper

    parser = argparse.ArgumentParser(
        description="Build a SQLite index from prot.accession2taxid.gz for fast lookups."
    )
    parser.add_argument(
        "accession2taxid_path", type=str,
        help="Path to prot.accession2taxid.gz",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output path for SQLite database (default: same directory as input)",
    )
    args = parser.parse_args(sys.argv[1:])

    mapper = AccessionTaxidMapper(args.accession2taxid_path)
    mapper.build_sqlite_index(args.output)


if __name__ == "__main__":
    main()
