import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIG ---
FEATURES = ["gdp_per_capita", "gdp_variation", "year"]
TARGET = "period_life_expectancy_at_birth"


# --- NODOS ---
def prepare_data_reg(merged_data: pd.DataFrame):
    """Prepara los datos para regresión."""
    df = merged_data.copy()

    # Asegurar columnas numéricas
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_rf_reg(X_train: pd.DataFrame, y_train: pd.Series):
    """Entrena un modelo RandomForest Regressor."""
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_split=8,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def evaluate_reg(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    """Evalúa el modelo y devuelve un DataFrame con métricas."""
    y_pred = model.predict(X_test)

    metrics = {
        "Modelo": "RandomForest (Regresión)",
        "R2": r2_score(y_test, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "params": str(model.get_params()),
    }

    # 🔧 CSVDataSet necesita DataFrame
    return pd.DataFrame([metrics])
