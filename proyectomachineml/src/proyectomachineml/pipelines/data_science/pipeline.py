from kedro.pipeline import Pipeline, node
from .nodes import train_model

def create_pipeline(**kwargs):
    return Pipeline([
        node(train_model, ["merged_data", "parameters"], ["ml_model", "train_score"])
    ])

