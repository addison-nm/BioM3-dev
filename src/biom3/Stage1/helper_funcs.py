from biom3.backend.device import setup_logger, print_gpu_utilization

logger = setup_logger(__name__)


def print_summary(result):
    logger.info("Time: %.2f", result.metrics['train_runtime'])
    logger.info("Samples/second: %.2f", result.metrics['train_samples_per_second'])
    print_gpu_utilization()


# def print_memory_usage():
#     process = psutil.Process(os.getpid())
#     memory_in_bytes = process.memory_info().rss
#     memory_in_megabytes = memory_in_bytes / (1024 ** 2)
#     #print(f"Memory used by this script: {memory_in_megabytes:.2f} MB")

#     return memory_in_megabytes






