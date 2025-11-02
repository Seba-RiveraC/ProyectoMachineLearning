from kedro.pipeline import Pipeline, node, pipeline
from .nodes import prepare_data_clf, train_rf_clf, evaluate_clf

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_data_clf,
            inputs="merged_data",
            outputs=["X_train_imp", "X_test_imp", "X_train_s", "X_test_s", "y_train_clf", "y_test_clf"],
            name="prepare_data_clf"
        ),
        node(
            func=train_rf_clf,
            inputs=["X_train_imp", "y_train_clf"],
            outputs="rf_clf_model",
            name="train_rf_clf"
        ),
        node(
            func=evaluate_clf,
            inputs=["rf_clf_model", "X_test_imp", "y_test_clf"],
            outputs="classification_metrics",
            name="evaluate_clf"
        ),
    ])
