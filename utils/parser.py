import os
from argparse import ArgumentParser, Namespace

from torch.utils.tensorboard import SummaryWriter

from utils import logging_utils
from utils.y_params import YParams


def get_parser() -> ArgumentParser:
    """
    Constructs and returns an ArgumentParser for command-line options.

    :return: Configured ArgumentParser.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--run_num",
        default='00',
        type=str,
        help='index of the current experiment'
    )
    parser.add_argument(
        "--yaml_config",
        default='./config/coursework_transformer.yaml',
        type=str,
        help='path to yaml file containing training configuration'
    )
    parser.add_argument(
        "--config",
        default='base',
        type=str,
        help='name of desired config in yaml file (base or short)'
    )
    parser.add_argument(
        "--num_iters",
        default=None,
        type=int,
        help='number of iters to run'
    )
    parser.add_argument(
        "--num_data_workers",
        default=None,
        type=int,
        help='number of data workers for data loader'
    )

    return parser


def parse_args() -> Namespace:
    """
    Parses command line arguments.

    :return: Namespace containing command line arguments.
    """
    return get_parser().parse_args()


def get_params(
        log_filename: str = 'out.log',
        logs_filepath: str = 'logs/'
) -> YParams:
    """
    Retrieves and prepares training parameters from a YAML configuration file, adjusting settings based on command line arguments.

    :param log_filename: Name of the file where logs should be written, defaults to 'out.log'.
    :param logs_filepath: Path relative to the experiment directory for storing TensorBoard logs, defaults to 'logs/'.
    :return: An instance of YParams containing all training and model parameters.
    """
    args: Namespace = parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Update config with modified args
    if args.num_iters:
        params.update({"num_iters": args.num_iters})

    if args.num_data_workers:
        params.update({"num_data_workers": args.num_data_workers})

    params.local_batch_size = params.global_batch_size

    # Set up directory
    base_dir: str = params.exp_dir
    exp_dir: str = os.path.join(base_dir, args.config + str(args.run_num) + '/')

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    logging_utils.log_to_file(
        logger_name=None,
        log_filename=os.path.join(exp_dir, log_filename)
    )
    params.log()

    args.tboard_writer = SummaryWriter(log_dir=os.path.join(exp_dir, logs_filepath))

    params.experiment_dir = os.path.abspath(exp_dir)

    return params
