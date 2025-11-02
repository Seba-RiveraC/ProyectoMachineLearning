# src/proyectomachineml/pipelines/data_engineering/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import merge_data

def create_pipeline(**kwargs) -> Pipeline:
    """
    Este pipeline se encarga de la fusión de datos ya limpios.
    """
    return Pipeline([
        node(
            func=merge_data,
            inputs=["gdp_clean", "life_clean"],
            outputs="merged_data",
            name="merge_cleaned_data_node",
        ),
    ])