import os
import argparse
import yaml
import shutil
from pathlib import Path
from datetime import datetime

from algos.vpg import VPGAlgo
from algos.vpg_dmp import (
    VPG_DMPAlgo,
    VPG_Policy_DMPAlgo,
    DoubleVPG_DMPAlgo,
    TargetVPG_DMPAlgo
)
import warnings


ALGO = {
    "vpg": VPGAlgo,
    "vpg_dmp": VPG_DMPAlgo,
    "vpg_policy_dmp": VPG_Policy_DMPAlgo,
    "double_vpg_dmp": DoubleVPG_DMPAlgo,
    "target_vpg_dmp": TargetVPG_DMPAlgo,
}


def dir_setup(dir_name, config_path):
    """
    Set up experiment and buffer directory. Either create with passed non None name or
    depending on configuration and current time. Create directory and save in it the
    config chosen, this is important because small changes might be made to a config
    before running.

    Args:
        dir_name (str): name of the directory where experiment will be saved
        config_path (str): path the config file
    Return:
        (str): full dir path of where to save experiment
    """
    if dir_name is not None:
        experiment_dir = dir_name
        if not "experiments" in dir_name:
            experiment_dir = "experiments/" + dir_name
    else:
        config_name = config_path.split("/")[1].split(".")[0]
        current_time = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        experiment_dir = "experiments/" + config_name + "_" + current_time
    buffer_dir = experiment_dir + "/buffer/"

    Path(experiment_dir).mkdir(parents=True, exist_ok=True)
    Path(buffer_dir).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, experiment_dir / Path(config_path.split("/")[1]))
    # os.system("git diff src/ > " + experiment_dir + "/diff")

    return experiment_dir, buffer_dir


def main():
    parser = argparse.ArgumentParser(description="Train settings.")
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        type=str,
        help="Path to config file or existing experiment.",
    )
    parser.add_argument(
        "-d",
        "--dir",
        required=False,
        type=str,
        help="Specify custom directory name to save experiment under experiments/ dir.",
    )
    parser.add_argument(
        "-lc",
        "--load-critic",
        required=False,
        type=str,
        help="Specify path to a critic model (.pth).",
    )
    parser.add_argument(
        "-lp",
        "--load-policy",
        required=False,
        type=str,
        help="Specify path to a policy model (.pth).",
    )
    parser.add_argument("-t", "--test", action="store_true", help="Testing flag.")
    args = parser.parse_args()
    assert not args.test or args.dir is not None, "Pass dir argument if testing."

    try:
        with open(args.path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    except OSError as e:
        print("Configuration file not available, check under configs/ directory.")
        print(e)
        sys.exit(1)
    experiment_dir, buffer_dir = dir_setup(args.dir, args.path)

    algo = ALGO[config["algo"]["name"]](config, experiment_dir, buffer_dir, args.test)

    if args.test:
        algo.test()
    else:
        algo.train(args.load_critic, args.load_policy)


if __name__ == "__main__":
    main()
