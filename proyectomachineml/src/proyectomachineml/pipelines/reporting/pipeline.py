from kedro.pipeline import Pipeline, node
from .nodes import plot_gdp_vs_life

def create_pipeline(**kwargs):
    return Pipeline([
        node(plot_gdp_vs_life, "merged_data", None)
    ])

