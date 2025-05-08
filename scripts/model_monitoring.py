import argparse
import configparser
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.housing.model_monitoring import run_monitoring

parser = argparse.ArgumentParser()

parser.add_argument(
    "train_data_path",
    type=str,
    help="description of arg1",
    nargs="?",
    default="data/processed/train/",
)
parser.add_argument(
    "test_data_path",
    type=str,
    help="description of arg2",
    nargs="?",
    default="data/processed/test/",
)

parser.add_argument(
    "report_path",
    type=str,
    help="description of arg3",
    nargs="?",
    default="reports/",
)

args = parser.parse_args()

config = configparser.ConfigParser()
config.read("setup.cfg")

run_monitoring(args)
