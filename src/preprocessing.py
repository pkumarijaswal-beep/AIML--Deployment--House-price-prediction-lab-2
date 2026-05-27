import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing


def load_and_split():

    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y=pd.Series(data.target)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )