# src/proyectomachineml/pipeline_registry.py
from typing import Dict
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from proyectomachineml.pipelines import data_cleaning
from kedro.pipeline import Pipeline
from .pipelines import regression, classification

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "regression": regression.create_pipeline(),
        "classification": classification.create_pipeline(),
        "__default__": regression.create_pipeline() + classification.create_pipeline(),
    }
