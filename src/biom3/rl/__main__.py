import sys


def run_grpo_train():
    from biom3.rl.run_grpo_train import main, parse_arguments
    args = parse_arguments(sys.argv[1:])
    main(args)


if __name__ == "__main__":
    run_grpo_train()
