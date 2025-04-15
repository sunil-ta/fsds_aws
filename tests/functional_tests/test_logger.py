import os
import sys



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.housing.logger import Logger

def test_logger():
    logger = Logger(
        "logs/test.log",
        "test succesfull",
        "w",
    )
    assert logger is not None
