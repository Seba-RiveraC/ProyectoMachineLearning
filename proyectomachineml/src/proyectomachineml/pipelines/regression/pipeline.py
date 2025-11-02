from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_data_reg, train_rf_reg, evaluate_reg

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_data_reg,
            inputs="merged_data",
            outputs=["X_train_reg", "X_test_reg", "y_train_reg", "y_test_reg"],
            name="prepare_data_reg"
        ),
        node(
            func=train_rf_reg,
            inputs=["X_train_reg", "y_train_reg"],
            outputs="rf_reg_model",
            name="train_rf_reg"
        ),
        node(
            func=evaluate_reg,
            inputs=["rf_reg_model", "X_test_reg", "y_test_reg"],
            outputs="regression_metrics",
            name="evaluate_reg"
        ),
    ])
