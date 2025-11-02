import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_gdp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame de GDP.
    
    Args:
        df: DataFrame original de GDP.
        
    Returns:
        DataFrame de GDP limpio y estandarizado.
    """
    logger.info("Limpiando dataset de GDP...")
    
    # Renombrar columnas para estandarizar
    df = df.rename(columns={'country_name': 'country', 'year': 'year', 'total_gdp_million': 'gdp_million'})
    
    # Eliminar filas con valores nulos o no válidos en 'gdp_million'
    df = df.dropna(subset=['gdp_million'])
    
    # Seleccionar solo las columnas de interés
    df = df[['country', 'year', 'gdp_million']]
    
    logger.info(f"Dataset de GDP limpio. Filas: {df.shape[0]}")
    return df

def clean_life_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el DataFrame de esperanza de vida.
    
    Args:
        df: DataFrame original de esperanza de vida.
        
    Returns:
        DataFrame de esperanza de vida limpio y estandarizado.
    """
    logger.info("Limpiando dataset de esperanza de vida...")
    
    # Renombrar columnas para estandarizar
    df = df.rename(columns={'Entity': 'country', 'Year': 'year', 'Period life expectancy at birth': 'life_expectancy'})
    
    # Eliminar filas con valores nulos o no válidos
    df = df.dropna(subset=['life_expectancy'])
    
    # Seleccionar solo las columnas de interés
    df = df[['country', 'year', 'life_expectancy']]
    
    logger.info(f"Dataset de esperanza de vida limpio. Filas: {df.shape[0]}")
    return df

def merge_data(gdp_cleaned: pd.DataFrame, life_cleaned: pd.DataFrame) -> pd.DataFrame:
    """
    Fusiona los datasets limpios de GDP y esperanza de vida.
    
    Args:
        gdp_cleaned: DataFrame de GDP limpio.
        life_cleaned: DataFrame de esperanza de vida limpio.
        
    Returns:
        DataFrame fusionado.
    """
    logger.info("Fusionando datasets de GDP y esperanza de vida...")
    merged_df = pd.merge(
        gdp_cleaned,
        life_cleaned,
        on=['country', 'year'],
        how='inner'  # Puedes cambiar a 'outer', 'left', 'right' según lo necesites
    )
    logger.info(f"Datasets fusionados. Filas totales: {merged_df.shape[0]}")
    return merged_df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza ingeniería de características en el DataFrame fusionado.
    
    Args:
        df: DataFrame fusionado.
        
    Returns:
        DataFrame con nuevas características.
    """
    logger.info("Realizando ingeniería de características...")
    
    # Ejemplo: Crear una nueva columna que combine GDP y esperanza de vida.
    df['gdp_per_capita'] = df['gdp_million'] / 1000 # Esto es solo un ejemplo
    
    logger.info("Ingeniería de características completada.")
    return df