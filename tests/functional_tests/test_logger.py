import os
import sys

from src.logger import Logger

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_logger():
    logger = Logger(
        "logs/test.log",
        "test succesfull",
        "w",
    )
    assert logger is not None
