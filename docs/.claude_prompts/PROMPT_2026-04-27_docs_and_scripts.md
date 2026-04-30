# April 27, 2026 todo items for biom3

I want to implement the following developments to biom3. You will need to start by familiarizing yourself with the project. Then, look over these items and construct a plan to accomplish each. The order we do things is flexible, but I think CLI and documentation related items should go last.

## Memory overload fix in run_PenCL_inference

In src/biom3/Stage2/run_Facilitator_sample.py we have an option --mmd_sample_limit that reduces the memory overhead. It's safe, since the results impacted by that setting are not saved, just printed. The problem is that in run_PenCL_inference.py we do not have a similar option, and see out of memory errors when a cross-datapoint comparison is made on large dataset sizes. I want to:

1. Verify that limiting the number used does not impact saved results.
2. Implement a similar limit argument. 

## Current default and optimal determination of limit_[train,val]_batches

We use by default an 80/20 train/validation split. For variable dataset sizes, it's unclear to me what the optimal configurations are for the pytorch lightning parameters limit_[train,val]_batches. Look into our current defaults, and reconcile whether these are optimal. Keep in mind that in some cases we expect to have only on the order of 10s of training epochs on very large datasets (10M plus entries) vs in other cases many (100s) of epochs, on smaller (~10K) dataset sizes.


## Argument check and CLI docs

We've made updates to training scripts, and we need to make sure our argparsers are correct, and that our documentation is accurate and with full coverage. Specifically, I want to do the following:

1. Check argparsers across all our entrypoints. Make sure each argument is accurately scoped and with a correct help statement.
2. Update the docstrings header at the top of each entrypoint's corresopnding .py script. Make sure all CLI args are accurately included, as well as any configuration files. 
3. Update the CLI_reference.md docs file with this information
4. Update any other relevant docs.
5. Update the top level README.md file.

## Script updates

Our changes need to be properly reflected in the jobs/ and scripts/ that we have. In particular, for stage3 (and 1,2) training/finetuning on aurora, polaris, and the spark, we need to make sure that we are properly calling the entrypoints. There was a major change made to how the wandb and wandb_api_key arguments are handled, and I want to make sure this is properly addressed in all scripts. Specifically, look at 

- jobs/aurora/_template*
- jobs/polaris/_template*
- jobs/spark/_template*
- scripts/launchers
- scripts/stage[1,2,3]_*.sh

As part of this review, also consider the machine specific settings we are using for aurora and polaris, implemented I think in environment.sh or in the scripts/launchers. I want to consider optimal configurations for a single-node job where all (e.g. 12 on aurora) devices are treated as effectively a single GPU. For example, for sequence generation jobs. Currently, we bind some number of CPU cores to each device. This may be suboptimal in some cases. 

## Refactor sync_databases.sh

The scripts/sync_databases.sh script is overly specific to "databases." (It differs from sync_weights in a subtle but important way.) Currently, users typically have to run `./scripts/sync_databases.sh </path/to/databases/src> </path/to/databases/tgt>` and then also `./scripts/sync_databases.sh </path/to/datasets/src> </path/to/datasets/tgt>`. It would be better to have a universal script `link_data.sh` that has the logic of sync_databases.sh, but uses this more appropriate name. To mirror, we should also use link_weights.sh, but again, note that there is a logical difference between the two scripts. In short:

- Rename scripts/sync_*.sh and modify script commentary to make data-source-agnostic
- maintain both weights and data versions.
- Document this change in the README and other relevant docs/

