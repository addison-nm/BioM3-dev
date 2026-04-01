"""NCBI taxonomy: accession-to-taxid mapping, lineage tree, and SQLite index."""

import gzip
import os
import sqlite3

import pandas as pd

from biom3.backend.device import setup_logger

logger = setup_logger(__name__)

RANKED_LINEAGE_COLUMNS = [
    "tax_id", "tax_name", "species", "genus", "family",
    "order", "class", "phylum", "kingdom", "superkingdom",
]

SQLITE_INDEX_FILENAME = "accession2taxid.sqlite"


def _parse_dmp_line(line):
    """Parse a pipe-delimited .dmp line into stripped fields."""
    return [f.strip() for f in line.rstrip("\t|\n").split("\t|\t")]


class TaxonomyTree:
    """In-memory NCBI taxonomy tree built from rankedlineage.dmp.

    Loads ~2.7M rows into a dict keyed by tax_id for O(1) lineage lookups.
    Memory footprint: ~300-500 MB.
    """

    def __init__(self, taxonomy_dir):
        """
        Args:
            taxonomy_dir: path to directory containing rankedlineage.dmp
                (and optionally nodes.dmp, names.dmp).
        """
        self.taxonomy_dir = taxonomy_dir
        self._lineages = {}
        self._loaded = False

    def load(self):
        """Parse rankedlineage.dmp into memory. Subsequent calls are no-ops."""
        if self._loaded:
            return
        lineage_path = os.path.join(self.taxonomy_dir, "rankedlineage.dmp")
        logger.info("Loading taxonomy from %s", lineage_path)
        count = 0
        with open(lineage_path, "r") as f:
            for line in f:
                fields = _parse_dmp_line(line)
                if len(fields) < len(RANKED_LINEAGE_COLUMNS):
                    fields.extend([""] * (len(RANKED_LINEAGE_COLUMNS) - len(fields)))
                tax_id = int(fields[0])
                self._lineages[tax_id] = {
                    col: fields[i]
                    for i, col in enumerate(RANKED_LINEAGE_COLUMNS)
                    if i > 0  # skip tax_id itself
                }
                count += 1
        self._loaded = True
        logger.info("Loaded %s taxonomy nodes", f"{count:,}")

    def get_lineage(self, tax_id):
        """Return ranked lineage for a tax_id as a dict.

        Keys: tax_name, species, genus, family, order, class, phylum,
        kingdom, superkingdom. Values are empty strings for missing ranks.
        """
        self._ensure_loaded()
        return self._lineages.get(tax_id, {})

    def get_lineage_string(self, tax_id):
        """Return formatted lineage string for text captions.

        Format: "The organism lineage is Bacteria, Pseudomonadota, ..."
        """
        lineage = self.get_lineage(tax_id)
        if not lineage:
            return None
        parts = []
        for rank in ["superkingdom", "kingdom", "phylum", "class",
                      "order", "family", "genus", "species"]:
            val = lineage.get(rank, "")
            if val:
                parts.append(val)
        if not parts:
            return None
        return "The organism lineage is " + ", ".join(parts)

    def filter_by_rank(self, tax_ids, rank, include=None, exclude=None):
        """Filter a set of tax_ids by taxonomic rank.

        Args:
            tax_ids: iterable of integer tax_ids.
            rank: rank name (e.g., 'superkingdom', 'phylum', 'family').
            include: if set, only keep tax_ids whose rank value is in this set.
            exclude: if set, remove tax_ids whose rank value is in this set.

        Returns:
            set of tax_ids that pass the filter.
        """
        self._ensure_loaded()
        result = set()
        for tid in tax_ids:
            lineage = self._lineages.get(tid, {})
            val = lineage.get(rank, "")
            if include and val not in include:
                continue
            if exclude and val in exclude:
                continue
            result.add(tid)
        return result

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()


class AccessionTaxidMapper:
    """Maps protein accessions to NCBI taxonomy IDs.

    The full prot.accession2taxid.gz has ~1.55B rows and cannot be loaded
    into memory. This class provides two lookup strategies:

    1. Streaming: reads the gzipped file in chunks, filtering to only
       requested accessions. No setup required, ~10-15 min per scan.
    2. SQLite index: one-time build creates an indexed database for
       instant lookups on subsequent calls.

    The lookup() method auto-detects which strategy to use.
    """

    def __init__(self, accession2taxid_path):
        """
        Args:
            accession2taxid_path: path to prot.accession2taxid.gz (or .tsv).
        """
        self.accession2taxid_path = accession2taxid_path

    def _default_sqlite_path(self):
        parent = os.path.dirname(self.accession2taxid_path)
        return os.path.join(parent, SQLITE_INDEX_FILENAME)

    def lookup(self, accessions, chunk_size=5_000_000):
        """Look up tax_ids for a set of accessions.

        Auto-detects: if a SQLite index exists at the conventional path,
        uses it; otherwise falls back to streaming.

        Args:
            accessions: iterable of accession strings.
            chunk_size: rows per chunk when streaming (ignored for SQLite).

        Returns:
            dict mapping accession -> tax_id (int).
        """
        sqlite_path = self._default_sqlite_path()
        if os.path.exists(sqlite_path):
            logger.info("SQLite index found, using fast lookup")
            return self.lookup_sqlite(accessions, sqlite_path)
        return self._lookup_streaming(accessions, chunk_size)

    def _lookup_streaming(self, accessions, chunk_size):
        """Stream the gzipped file in chunks, filtering to requested accessions."""
        accession_set = set(accessions)
        results = {}
        path = self.accession2taxid_path
        logger.info("Streaming accession2taxid lookup for %s accessions from %s",
                     f"{len(accession_set):,}", path)

        is_gzipped = path.endswith(".gz")
        opener = gzip.open if is_gzipped else open
        mode = "rt" if is_gzipped else "r"

        rows_scanned = 0
        for chunk in pd.read_csv(
            path, sep="\t", chunksize=chunk_size,
            usecols=["accession", "taxid"],
            compression="gzip" if is_gzipped else None,
        ):
            match = chunk[chunk["accession"].isin(accession_set)]
            for _, row in match.iterrows():
                results[row["accession"]] = int(row["taxid"])
            rows_scanned += len(chunk)
            if rows_scanned % (chunk_size * 10) == 0:
                logger.info("  Scanned %s rows, found %s/%s accessions",
                             f"{rows_scanned:,}", f"{len(results):,}",
                             f"{len(accession_set):,}")
            # Early exit if all found
            if len(results) == len(accession_set):
                logger.info("  All accessions found after %s rows", f"{rows_scanned:,}")
                break

        logger.info("Lookup complete: %s/%s accessions found",
                     f"{len(results):,}", f"{len(accession_set):,}")
        return results

    def build_sqlite_index(self, output_path=None):
        """Build a SQLite index from prot.accession2taxid.gz.

        One-time operation. Streams the full file into a SQLite DB with
        an index on the accession column.

        Args:
            output_path: where to write the .sqlite file. Defaults to
                the same directory as the source file.
        """
        output_path = output_path or self._default_sqlite_path()
        path = self.accession2taxid_path
        is_gzipped = path.endswith(".gz")
        logger.info("Building SQLite index at %s from %s", output_path, path)

        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS accession2taxid "
            "(accession TEXT PRIMARY KEY, taxid INTEGER)"
        )
        conn.commit()

        rows_inserted = 0
        for chunk in pd.read_csv(
            path, sep="\t", chunksize=5_000_000,
            usecols=["accession", "taxid"],
            compression="gzip" if is_gzipped else None,
        ):
            chunk.to_sql(
                "accession2taxid", conn,
                if_exists="append", index=False,
                method="multi",
            )
            rows_inserted += len(chunk)
            if rows_inserted % 50_000_000 == 0:
                logger.info("  Inserted %s rows", f"{rows_inserted:,}")

        logger.info("Creating index on accession column...")
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_accession "
            "ON accession2taxid(accession)"
        )
        conn.commit()
        conn.close()
        logger.info("SQLite index built: %s rows at %s",
                     f"{rows_inserted:,}", output_path)

    def lookup_sqlite(self, accessions, db_path=None):
        """Fast lookup via pre-built SQLite index.

        Args:
            accessions: iterable of accession strings.
            db_path: path to the SQLite database. Defaults to conventional path.

        Returns:
            dict mapping accession -> tax_id (int).
        """
        db_path = db_path or self._default_sqlite_path()
        accession_list = list(accessions)
        conn = sqlite3.connect(db_path)
        results = {}

        # Query in batches of 500 to avoid SQLite variable limit
        batch_size = 500
        for i in range(0, len(accession_list), batch_size):
            batch = accession_list[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            cursor = conn.execute(
                f"SELECT accession, taxid FROM accession2taxid "
                f"WHERE accession IN ({placeholders})",
                batch,
            )
            for acc, taxid in cursor:
                results[acc] = taxid

        conn.close()
        logger.info("SQLite lookup: %s/%s accessions found",
                     f"{len(results):,}", f"{len(accession_list):,}")
        return results
