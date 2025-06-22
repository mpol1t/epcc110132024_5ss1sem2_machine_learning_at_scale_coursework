import logging
import os

import torch.distributed as dist

_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


class GlobalRankFilter(logging.Filter):
    """
    A logging filter that only allows log records from the global rank 0 process.
    If torch.distributed is initialized, it uses `dist.get_rank()`; otherwise, it falls
    back to checking the LOCAL_RANK environment variable.
    """

    def filter(self, record):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
        return os.environ.get("LOCAL_RANK", "0") == "0"


def config_logger(log_level: int = logging.INFO) -> None:
    """
    Configures the root logger with a specific format and log level.
    Adds GlobalRankFilter so that only the global rank 0 process logs messages.

    :param log_level: Level of logging to capture, defaults to logging.INFO.
    """
    logging.basicConfig(format=_FORMAT, level=log_level)
    root_logger = logging.getLogger()
    # Ensure that only rank 0 emits log messages.
    root_logger.addFilter(GlobalRankFilter())


def log_to_file(logger_name: None = None, log_level: int = logging.INFO, log_filename: str = 'tensorflow.log') -> None:
    """
    Configures logging to a file with a specific log level and filename.
    Creates the directory for the log file if it doesn't exist.
    Adds GlobalRankFilter to the file handler to avoid duplicate logs in a distributed setting.

    :param logger_name: Name of the logger to configure; defaults to the root logger if None.
    :param log_level: Level of logging to capture, defaults to logging.INFO.
    :param log_filename: Filename for the log file, defaults to 'tensorflow.log'.
    """
    directory = os.path.dirname(log_filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    if logger_name is not None:
        log = logging.getLogger(logger_name)
    else:
        log = logging.getLogger()

    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_level)
    fh.setFormatter(logging.Formatter(_FORMAT))
    # Add the GlobalRankFilter to the file handler as well
    fh.addFilter(GlobalRankFilter())
    log.addHandler(fh)


def log_versions():
    """
    Logs version information for the current Git branch, Git commit hash, and PyTorch version.
    This log will only appear if the process is the global (rank 0) process.
    """
    import torch
    import subprocess

    logging.info('--------------- Versions ---------------')
    logging.info('git branch: ' + str(subprocess.check_output(['git', 'branch']).strip()))
    logging.info('git hash: ' + str(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()))
    logging.info('Torch: ' + str(torch.__version__))
    logging.info('----------------------------------------')
