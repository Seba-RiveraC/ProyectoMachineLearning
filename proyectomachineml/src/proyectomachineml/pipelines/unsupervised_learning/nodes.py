"""
Nodos para aprendizaje no supervisado: preparación, clustering y reducción de dimensionalidad.
"""

from __future__ import annotations
import pandas as pd
from typing import Dict, Tuple, Any
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

try:
    import umap  # type: ignore
    UMAP_AVAILABLE = True
except ImportError:
    umap = None  # type: ignore
    UMAP_AVAILABLE = False

def select_and_encode_features(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    """Escala variables numéricas y codifica variables categóricas."""
    transformer = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    X_processed = transformer.fit_transform(df)
    cat_features = transformer.named_transformers_["cat"].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_features)
    return pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        columns=feature_names,
        index=df.index,
    )

def kmeans_clustering(X: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Ejecuta K-Means y devuelve etiquetas y métricas."""
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    metrics: Dict[str, Any] = {
        "silhouette_score": float(silhouette_score(X, labels)) if len(set(labels)) > 1 else None,
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)) if len(set(labels)) > 1 else None,
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)) if len(set(labels)) > 1 else None,
    }
    result_df = X.copy()
    result_df["cluster_label"] = labels
    return result_df, metrics

def dbscan_clustering(X: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Ejecuta DBSCAN y devuelve etiquetas (incluyendo ruido) y métricas."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    unique_labels = set(labels)
    if len(unique_labels - {-1}) <= 1:
        metrics = {"silhouette_score": None, "davies_bouldin_score": None, "calinski_harabasz_score": None}
    else:
        mask = labels != -1
        metrics = {
            "silhouette_score": float(silhouette_score(X[mask], labels[mask])),
            "davies_bouldin_score": float(davies_bouldin_score(X[mask], labels[mask])),
            "calinski_harabasz_score": float(calinski_harabasz_score(X[mask], labels[mask])),
        }
    result_df = X.copy()
    result_df["cluster_label"] = labels
    return result_df, metrics

def agglomerative_clustering(X: pd.DataFrame, n_clusters: int = 3, linkage: str = "ward") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Ejecuta clustering jerárquico aglomerativo y devuelve etiquetas y métricas."""
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X)
    metrics = {
        "silhouette_score": float(silhouette_score(X, labels)) if len(set(labels)) > 1 else None,
        "davies_bouldin_score": float(davies_bouldin_score(X, labels)) if len(set(labels)) > 1 else None,
        "calinski_harabasz_score": float(calinski_harabasz_score(X, labels)) if len(set(labels)) > 1 else None,
    }
    result_df = X.copy()
    result_df["cluster_label"] = labels
    return result_df, metrics

def perform_pca(X: pd.DataFrame, n_components: int = 2) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Aplica PCA y devuelve la proyección y la varianza explicada."""
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X)
    comp_names = [f"PC{i+1}" for i in range(components.shape[1])]
    comp_df = pd.DataFrame(components, columns=comp_names, index=X.index)
    metrics = {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
    }
    return comp_df, metrics

def perform_umap(X: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42) -> pd.DataFrame:
    """Aplica UMAP para reducción de dimensionalidad. Requiere instalar `umap-learn`."""
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP no está disponible. Instala 'umap-learn' para utilizar esta función.")
    mapper = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = mapper.fit_transform(X)
    columns = [f"UMAP{i+1}" for i in range(n_components)]
    return pd.DataFrame(embedding, columns=columns, index=X.index)
