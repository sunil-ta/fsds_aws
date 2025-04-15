import os
import pickle
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_main():
    model_file = os.getcwd() + "/artifacts/model/" + "lin_reg.pkl"
    with open(model_file, "rb") as f:
        model = pickle.load(f)
        assert hasattr(model, "predict")
