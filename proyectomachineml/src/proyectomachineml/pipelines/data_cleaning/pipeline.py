# src/proyectomachineml/pipelines/data_cleaning/pipeline.py
from kedro.pipeline import Pipeline, node
from .nodes import clean_gdp_data, clean_life_data, merge_data, feature_engineering

def create_pipeline(**kwargs) -> Pipeline:
    """
    Crea el pipeline de ingeniería de datos.
    
    Args:
        kwargs: Parámetros del proyecto.
        
    Returns:
        El pipeline completo.
    """
    return Pipeline(
        [
            # Nodo para limpiar los datos de GDP
            node(
                func=clean_gdp_data,
                inputs="countriesgdphist",
                outputs="gdp_cleaned",
                name="clean_gdp_data_node",
            ),
            
            # Nodo para limpiar los datos de esperanza de vida
            node(
                func=clean_life_data,
                inputs="lifeexpectancy",
                outputs="life_cleaned",
                name="clean_life_data_node",
            ),
            
            # Nodo para fusionar los datasets limpios
            node(
                func=merge_data,
                inputs=["gdp_cleaned", "life_cleaned"],
                outputs="merged_data",
                name="merge_data_node",
            ),
            
            # Nodo para realizar ingeniería de características (opcional)
            node(
                func=feature_engineering,
                inputs="merged_data",
                outputs="engineered_features",
                name="feature_engineering_node",
            )
        ]
    )