import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(merged_data: pd.DataFrame, params: dict):
    X = merged_data[["total_gdp"]]
    y = merged_data["life_expectancy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params["train"]["test_size"], random_state=params["train"]["random_state"]
    )
    model = LinearRegression().fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score

