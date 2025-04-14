import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_installation():
    try:
        import src
    except ImportError:
        assert False, "Failed to import the package src"
    else:
        assert True


if __name__ == "__main__":
    pass
