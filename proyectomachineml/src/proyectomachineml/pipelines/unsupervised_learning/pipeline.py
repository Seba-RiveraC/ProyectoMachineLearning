"""
Pipeline para aprendizaje no supervisado.

Encadena el preprocesamiento de características, tres algoritmos de clustering
(K-Means, DBSCAN y aglomerativo) y la reducción de dimensionalidad (PCA y UMAP).
"""

from __future__ import annotations
from kedro.pipeline import Pipeline, node
from .nodes import (
    select_and_encode_features,
    kmeans_clustering,
    dbscan_clustering,
    agglomerative_clustering,
    perform_pca,
    perform_umap,
)

def create_pipeline(**kwargs) -> Pipeline:
    """Construye la pipeline de aprendizaje no supervisado."""
    return Pipeline(
        [
            node(
                select_and_encode_features,
                inputs=dict(
                    df="master_table",
                    numeric_cols="params:unsupervised.numerical_columns",
                    categorical_cols="params:unsupervised.categorical_columns",
                ),
                outputs="unsupervised_preprocessed",
                name="preprocess_for_unsupervised",
            ),
            node(
                kmeans_clustering,
                inputs=dict(
                    X="unsupervised_preprocessed",
                    n_clusters="params:unsupervised.kmeans.n_clusters",
                    random_state="params:unsupervised.kmeans.random_state",
                ),
                outputs=["kmeans_clustered", "kmeans_metrics"],
                name="run_kmeans",
            ),
            node(
                dbscan_clustering,
                inputs=dict(
                    X="unsupervised_preprocessed",
                    eps="params:unsupervised.dbscan.eps",
                    min_samples="params:unsupervised.dbscan.min_samples",
                ),
                outputs=["dbscan_clustered", "dbscan_metrics"],
                name="run_dbscan",
            ),
            node(
                agglomerative_clustering,
                inputs=dict(
                    X="unsupervised_preprocessed",
                    n_clusters="params:unsupervised.agglomerative.n_clusters",
                    linkage="params:unsupervised.agglomerative.linkage",
                ),
                outputs=["agg_clustered", "agg_metrics"],
                name="run_agglomerative",
            ),
            node(
                perform_pca,
                inputs=dict(
                    X="unsupervised_preprocessed",
                    n_components="params:unsupervised.pca.n_components",
                ),
                outputs=["pca_embedding", "pca_metrics"],
                name="run_pca",
            ),
            node(
                perform_umap,
                inputs=dict(
                    X="unsupervised_preprocessed",
                    n_components="params:unsupervised.umap.n_components",
                    n_neighbors="params:unsupervised.umap.n_neighbors",
                    min_dist="params:unsupervised.umap.min_dist",
                    random_state="params:unsupervised.umap.random_state",
                ),
                outputs="umap_embedding",
                name="run_umap",
            ),
        ]
    )
