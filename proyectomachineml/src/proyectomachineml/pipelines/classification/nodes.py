import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

FEATURES = ["gdp_per_capita", "gdp_variation", "year"]
TARGET_CONT = "life_expectancy"


def prepare_data_clf(merged_data: pd.DataFrame):
    """Prepara datos para clasificación."""
    df = merged_data.copy()

    # Asegurar columnas numéricas
    for col in FEATURES + [TARGET_CONT]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURES + [TARGET_CONT])

    threshold = df[TARGET_CONT].median()
    df["high_life_exp"] = (df[TARGET_CONT] >= threshold).astype(int)

    X = df[FEATURES]
    y = df["high_life_exp"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train)
    X_test_imp = imp.transform(X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_test_s = scaler.transform(X_test_imp)

    return X_train_imp, X_test_imp, X_train_s, X_test_s, y_train, y_test


def train_rf_clf(X_train_imp, y_train):
    """Entrena un RandomForestClassifier."""
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1
    )
    rf.fit(X_train_imp, y_train)
    return rf

def evaluate_clf(model: RandomForestClassifier, X_test_imp, y_test):
    """Evalúa el modelo y devuelve un dict JSON-serializable."""
    proba = model.predict_proba(X_test_imp)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "Modelo": "RandomForest (Clasificación)",
        "Accuracy": float(accuracy_score(y_test, pred)),
        "Precision": float(precision_score(y_test, pred)),
        "Recall": float(recall_score(y_test, pred)),
        "F1": float(f1_score(y_test, pred)),
        "ROC_AUC": float(roc_auc_score(y_test, proba)),
        "params": {k: str(v) for k, v in model.get_params().items()}
    }

    # ← DEVOLVEMOS DICCIONARIO, NO DATAFRAME
    return metrics



