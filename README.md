Predicción de Esperanza de Vida según PIB — Proyecto de Machine Learning

Este proyecto implementa un pipeline completo de ciencia de datos usando Kedro, DVC, Docker y Airflow, para analizar la relación entre indicadores económicos (como el PIB per cápita) y la esperanza de vida. Incluye modelos supervisados, no supervisados y visualizaciones diseñadas para entregar resultados claros y totalmente reproducibles.

Objetivos del Proyecto

Analizar cómo el PIB per cápita y otras variables influyen en la esperanza de vida.

Implementar modelos de regresión, clasificación y clustering.

Automatizar flujos mediante Airflow.

Contener y aislar el entorno mediante Docker para asegurar reproducibilidad.

Tecnologías utilizadas

Kedro: arquitectura modular del pipeline

DVC: versionado de datos, métricas y modelos

Airflow: orquestación de tareas del pipeline

Docker: contenedorización del entorno de ejecución

Scikit-learn: modelos de aprendizaje automático

Pandas, NumPy y matplotlib: análisis y visualización

UMAP y KMeans: análisis no supervisado

Jupyter Notebook: informe final de presentación

Estructura del Proyecto
PROYECTOMACHINELEARNING/
├── .dvc/
├── .gitignore
├── dockerfile
├── dvc.yaml
├── docker-compose.yaml        # Integración con Airflow (orquestación)
├── airflow/                   # DAGs y configuración
│   ├── dags/
│   └── logs/
├── pyproject.toml
├── README.md
│
├── data/
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   ├── 04_feature/
│   ├── 05_model_input/
│   ├── 06_models/
│   ├── 07_model_output/
│   │   ├── best_classification_model.pkl
│   │   ├── best_regression_model.pkl
│   │   ├── classification_metrics.json
│   │   ├── dbscan_clustered.csv
│   │   ├── kmeans_clustered.csv
│   │   ├── pca_embedding.csv
│   │   ├── regression_metrics.json
│   │   └── umap_embedding.csv
│   └── 08_reporting/
│
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_data_understanding.ipynb
│   ├── analysis_report.ipynb
│   └── Presentacion.ipynb
│
├── src/
│   ├── proyectomachinelearning/
│   │   ├── __init__.py
│   │   ├── pipeline_registry.py
│   │   ├── settings.py
│   │   │
│   │   ├── pipelines/
│   │   │   ├── data_cleaning/
│   │   │   ├── data_engineering/
│   │   │   ├── data_science/
│   │   │   ├── regression/
│   │   │   ├── reporting/
│   │   │   └── unsupervised_learning/
│   │   │
│   │   └── tests/
│   └── requirements.txt
│
└── venv/   (no incluido en el repositorio)

¿Cómo ejecutar este proyecto?
1. Crear entorno virtual
python -m venv venv

2. Activar entorno virtual (Windows)
.\venv\Scripts\activate

3. Instalar dependencias
pip install -r src/requirements.txt

4. Ejecutar el pipeline completo con Kedro
kedro run

5. (Opcional) Ejecutar pipeline orquestado con Airflow
docker-compose up -d


Los DAGs se cargan desde:

airflow/dags/

6. Obtener datos y modelos versionados con DVC
dvc pull

7. Ejecutar notebook de defensa
jupyter notebook notebooks/Defensa_Final_Presentacion.ipynb

Resultados destacados

Clasificación (Random Forest):
ROC-AUC ≈ 0.906

Regresión (Random Forest):
R² ≈ 0.636
RMSE ≈ 6.87
MAE ≈ 5.14

Clustering (UMAP + KMeans):
Agrupación clara de países según PIB y esperanza de vida.

¿Qué incluye este repositorio?

Limpieza y preparación de datos.

Modelado supervisado (clasificación y regresión).

Modelado no supervisado (UMAP + KMeans, DBSCAN).

Comparación de métricas y visualizaciones.

Reproducibilidad con Kedro, DVC y Docker.

Integración de flujo automatizado mediante Airflow.

notebooks/Presentacion.ipynb

Reproducibilidad

El pipeline completo puede ejecutarse con:

kedro run


Los datos, modelos y métricas están controlados por DVC.

Los flujos automatizados están disponibles como DAGs de Airflow.

La contenedorización mediante Docker asegura uniformidad entre entornos.

Los artefactos se organizan por etapas en la carpeta /data.
