import argparse
import configparser
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ingest_data import ingest

parser = argparse.ArgumentParser()
parser.add_argument(
    "download",
    type=str,
    help="description of arg1",
    nargs="?",
    default="data",
)
args = parser.parse_args()
config = configparser.ConfigParser()
config.read("setup.cfg")

arg1 = args.download
ingest(arg1)
