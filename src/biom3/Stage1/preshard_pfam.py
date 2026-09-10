"""Write a Stage 1 run's Pfam shards ahead of time, on one node.

Training's prepare_data writes one shard per rank on global rank 0 while every
other rank waits: ~87 min at 3,072 ranks on the full data (13 min of CSV load,
then ~1.28 s per shard file), on every job. The shards depend only on the Pfam
CSV and the world size, so they can be written once here and reused by every
job at that world size; prepare_data skips when the manifest matches.

    python -m biom3.Stage1.preshard_pfam -c CONFIG [--world_size W]

Settings come from the trainer's own parser, so the Pfam path, filters and
pfam_splits_dir are exactly what training will use. World size defaults to
num_nodes * devices_per_node from the config. The config must name a
persistent pfam_splits_dir; the per-run default would be written and never
reused.
"""
import argparse
import logging
import sys

from biom3.Stage1.preprocess import (
    Pfam_DataModule, pfam_splits_status, write_pfam_splits,
)
from biom3.Stage1.run_PL_training import retrieve_all_args

_RUN_ID = "__preshard__"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-c", "--config_path", required=True)
    ap.add_argument("--world_size", type=int, default=None,
                    help="default: num_nodes * devices_per_node from the config")
    ns, rest = ap.parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    args = retrieve_all_args(["--config_path", ns.config_path, "--run_id", _RUN_ID] + rest)
    world = ns.world_size or int(args.num_nodes) * int(args.devices_per_node)
    dm = Pfam_DataModule(args)
    splits_dir = dm._resolve_splits_dir()
    if _RUN_ID in splits_dir:
        sys.exit(f"pfam_splits_dir resolved to a per-run directory ({splits_dir}); "
                 "set a persistent pfam_splits_dir in the config")

    print(f"pfam_data_path  {args.pfam_data_path}\npfam_splits_dir {splits_dir}\nworld_size      {world}")
    if pfam_splits_status(splits_dir, args.pfam_data_path, world) == "reuse":
        print("complete, matching shards already present; nothing to do")
        return 0
    write_pfam_splits(dm.load_pfam_database(), splits_dir, world, args.pfam_data_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
