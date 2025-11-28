"""
Funciones de limpieza, combinación y generación de características para el proyecto.

Este módulo contiene nodos para:

* Cargar y limpiar los datos de PIB per cápita de países y organizaciones.
* Limpiar el dataset de esperanza de vida.
* Combinar los datasets de PIB y esperanza de vida en un único DataFrame.
* Generar características adicionales que sirvan como entrada al modelado supervisado y no supervisado.

Cada función se implementa como un nodo de Kedro y debe ser utilizada dentro
de la pipeline ``data_cleaning`` definida en ``pipeline.py``.
"""

from __future__ import annotations

import pandas as pd

def clean_gdp_countries(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset de PIB per cápita de países."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "countryname": "country_name",
        "countrycode": "country_code",
        "pib_per_capita": "gdp_per_capita",
        "pib_per_capita_(usd)": "gdp_per_capita",
    }
    df = df.rename(columns=rename_map)
    expected_cols = {"country_code", "year", "gdp_per_capita"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise KeyError(f"Las columnas {missing} son necesarias en el dataset de PIB per cápita de países.")
    # Normaliza códigos y nombres de país
    df["country_code"] = df["country_code"].astype(str).str.strip().str.upper()
    df["country_name"] = df.get("country_name", "").astype(str).str.strip()
    # Convierte a numérico
    df["gdp_per_capita"] = pd.to_numeric(df["gdp_per_capita"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df.dropna(subset=["country_code", "year", "gdp_per_capita"])
    return df

def clean_gdp_organizations(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset de PIB per cápita de organizaciones internacionales."""
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "countryname": "country_name",
        "countrycode": "country_code",
        "pib_per_capita": "gdp_per_capita",
        "pib_per_capita_(usd)": "gdp_per_capita",
    }
    df = df.rename(columns=rename_map)
    expected_cols = {"country_code", "year", "gdp_per_capita"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise KeyError(f"Las columnas {missing} son necesarias en el dataset de PIB per cápita de organizaciones.")
    # Normaliza códigos y nombres de país
    df["country_code"] = df["country_code"].astype(str).str.strip().str.upper()
    df["country_name"] = df.get("country_name", "").astype(str).str.strip()
    df["gdp_per_capita"] = pd.to_numeric(df["gdp_per_capita"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df = df.dropna(subset=["country_code", "year", "gdp_per_capita"])
    return df

def combine_gdp_datasets(countries_df: pd.DataFrame, org_df: pd.DataFrame) -> pd.DataFrame:
    """Combina los datasets de PIB per cápita y calcula la variación anual."""
    combined = pd.concat([countries_df, org_df], ignore_index=True)
    combined = combined.sort_values(["country_code", "year"])
    combined["gdp_variation"] = combined.groupby("country_code")["gdp_per_capita"].diff().fillna(0.0)
    return combined

def clean_life_expectancy(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset de esperanza de vida.

    Acepta variantes como 'Entity', 'Code' y 'Period life expectancy at birth'
    y las renombra a 'country_name', 'country_code' y 'life_expectancy'.
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rename_map = {
        "entity": "country_name",
        "code": "country_code",
        "countryname": "country_name",
        "countrycode": "country_code",
        "life_expectancy_at_birth": "life_expectancy",
        "period_life_expectancy_at_birth": "life_expectancy",
        "esperanza_de_vida": "life_expectancy",
    }
    df = df.rename(columns=rename_map)
    expected_cols = {"country_code", "year", "life_expectancy"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise KeyError(f"Las columnas {missing} son necesarias en el dataset de esperanza de vida.")
    df["life_expectancy"] = pd.to_numeric(df["life_expectancy"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["country_code"] = df["country_code"].astype(str).str.strip().str.upper()
    if "country_name" in df.columns:
        df["country_name"] = df["country_name"].astype(str).str.strip()
    df = df.dropna(subset=["country_code", "year", "life_expectancy"])
    return df

def merge_datasets(gdp_df: pd.DataFrame, life_df: pd.DataFrame) -> pd.DataFrame:
    """Une PIB y esperanza de vida en un DataFrame."""
    merged = pd.merge(
        gdp_df,
        life_df,
        on=["country_code", "year"],
        how="inner",
        suffixes=("_gdp", "_life"),
    )
    return merged

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Genera características adicionales (ingreso y normalizaciones)."""
    engineered = df.copy()
    gdp_quantiles = engineered["gdp_per_capita"].quantile([0.25, 0.5, 0.75])
    def income_group(gdp: float) -> str:
        if gdp < gdp_quantiles[0.25]:
            return "low_income"
        elif gdp < gdp_quantiles[0.5]:
            return "lower_middle_income"
        elif gdp < gdp_quantiles[0.75]:
            return "upper_middle_income"
        else:
            return "high_income"
    engineered["income_group"] = engineered["gdp_per_capita"].apply(income_group)
    for col in ["gdp_per_capita", "life_expectancy", "gdp_variation"]:
        min_val, max_val = engineered[col].min(), engineered[col].max()
        engineered[f"{col}_norm"] = 0.0 if max_val == min_val else (engineered[col] - min_val) / (max_val - min_val)
    return engineered
