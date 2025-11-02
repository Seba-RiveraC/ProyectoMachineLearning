# src/proyectomachineml/pipelines/data_engineering/nodes.py
import pandas as pd
import logging

def merge_data(gdp_clean: pd.DataFrame, life_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Fusiona los datasets de GDP y de esperanza de vida, que ya han sido limpiados.
    """
    logger = logging.getLogger(__name__)
    logger.info("Fusionando datasets limpios...")
    
    merged_df = pd.merge(
        gdp_clean,
        life_clean,
        left_on=['country_name', 'year'],
        right_on=['entity', 'year'],
        how='inner'
    )
    logger.info(f"Fusión completada. Filas finales: {len(merged_df)}")
    return merged_df