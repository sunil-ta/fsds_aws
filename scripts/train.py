import argparse
import configparser
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train import train

parser = argparse.ArgumentParser()
parser.add_argument(
    "train_data_path",
    type=str,
    help="description of arg1",
    nargs="?",
    default="data/processed/train/",
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

arg1 = args.train_data_path
arg2 = args.stored_model_path
train(arg1, arg2)
