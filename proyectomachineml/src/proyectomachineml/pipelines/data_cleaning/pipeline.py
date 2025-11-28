"""
Pipeline de limpieza y preparación de datos.

Esta pipeline lee los datasets crudos definidos en el catálogo, los limpia,
los combina y genera una tabla maestra lista para el modelado.
"""

from __future__ import annotations
from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_gdp_countries,
    clean_gdp_organizations,
    combine_gdp_datasets,
    clean_life_expectancy,
    merge_datasets,
    feature_engineering,
)

def create_pipeline(**kwargs) -> Pipeline:
    """Construye la pipeline de limpieza."""
    return Pipeline(
        [
            node(clean_gdp_countries, "pib_per_capita_countries", "gdp_countries_clean", name="clean_gdp_countries"),
            node(clean_gdp_organizations, "pib_per_capita_organizations", "gdp_organizations_clean", name="clean_gdp_organizations"),
            node(combine_gdp_datasets, ["gdp_countries_clean", "gdp_organizations_clean"], "gdp_cleaned", name="combine_gdp_datasets"),
            node(clean_life_expectancy, "life_expectancy", "life_cleaned", name="clean_life_expectancy"),
            node(merge_datasets, ["gdp_cleaned", "life_cleaned"], "merged_data", name="merge_datasets"),
            node(feature_engineering, "merged_data", "master_table", name="feature_engineering"),
        ]
    )
