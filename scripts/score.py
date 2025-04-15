import argparse
import configparser
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.housing.score import score

parser = argparse.ArgumentParser()
parser.add_argument(
    "test_data_path",
    type=str,
    help="description of arg1",
    nargs="?",
    default="data/processed/test",
)
parser.add_argument(
    "stored_model_path",
    type=str,
    help="description of arg2",
    nargs="?",
    default="artifacts/model",
)
args = parser.parse_args()

config = configparser.ConfigParser()
config.read("setup.cfg")

arg1 = args.test_data_path or config["DEFAULT"]["test_data_path"]
arg2 = args.stored_model_path or config["DEFAULT"]["stored_model_path"]
score(arg1, arg2)
