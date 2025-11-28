# src/proyectomachineml/pipeline_registry.py

from typing import Dict
from kedro.pipeline import Pipeline

from .pipelines import data_cleaning, regression, classification
from .pipelines.unsupervised_learning import pipeline as unsupervised_pipeline

def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "data_cleaning": data_cleaning.create_pipeline(),
        "regression": regression.create_pipeline(),
        "classification": classification.create_pipeline(),
        "unsupervised_learning": unsupervised_pipeline.create_pipeline(),
        "__default__": (
            data_cleaning.create_pipeline()
            + regression.create_pipeline()
            + classification.create_pipeline()
            + unsupervised_pipeline.create_pipeline()
        ),
    }
