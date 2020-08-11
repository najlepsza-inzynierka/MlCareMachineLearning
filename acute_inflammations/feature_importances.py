import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from data_loaders import load_acute_inflammations

if __name__ == "__main__":
    X, y = load_acute_inflammations()

