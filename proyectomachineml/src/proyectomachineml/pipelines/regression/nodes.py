import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Columnas correctas del master_table
FEATURES = ["gdp_per_capita", "gdp_variation", "year"]

# Target correcto
TARGET = "life_expectancy"


def prepare_data_reg(merged_data: pd.DataFrame):
    """Prepara datos para regresión."""
    df = merged_data.copy()

    # Convertir a numérico
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar filas incompletas
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Imputación
    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)

    # Escalamiento
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s = scaler.transform(X_test_imp)

    return X_train_s, X_test_s, y_train, y_test


def train_rf_reg(X_train_s, y_train):
    """Entrena un modelo RandomForest para regresión."""
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    return rf


def evaluate_reg(model: RandomForestRegressor, X_test_s, y_test):
    """Evalúa un modelo de regresión."""
    pred = model.predict(X_test_s)

    return {
        "Modelo": "RandomForestRegressor",
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mean_squared_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "R2": r2_score(y_test, pred),
        "params": str(model.get_params())
    }
