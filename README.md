#  Predicción de Esperanza de Vida según PIB — Proyecto de Machine Learning

Este proyecto implementa un pipeline completo de ciencia de datos usando **Kedro** y **DVC**, para analizar la relación entre indicadores económicos (como el PIB per cápita) y la esperanza de vida. Incluye modelos supervisados, no supervisados y visualizaciones para presentar resultados de forma clara y reproducible.

---

##  Objetivos del Proyecto

- Investigar cómo el PIB per cápita y otras variables explican la esperanza de vida.
- Aplicar técnicas de aprendizaje supervisado (regresión y clasificación) y no supervisado (clustering).
- Asegurar la reproducibilidad con herramientas profesionales como Kedro y DVC.
- Presentar resultados en un cuaderno final de defensa.

---

##  Tecnologías utilizadas

- **Kedro** → gestión modular del pipeline de ML
- **DVC** → versionado de datos, métricas y modelos
- **scikit-learn** → modelos de ML
- **Pandas & matplotlib** → análisis y visualización
- **UMAP & KMeans** → clustering y reducción de dimensionalidad
- **Jupyter Notebook** → presentación final integrada

PROYECTOMACHINELEARNING/
├── .dvc/
├── .gitignore
├── dockerfile
├── dvc.yaml
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

2. Activar entorno virtual

En Windows:

.\venv\Scripts\activate

3. Instalar dependencias
pip install -r src/requirements.txt

4. Ejecutar el pipeline de Kedro
kedro run

5. (Opcional) Obtener datos y modelos con DVC
dvc pull

6. Ejecutar notebook de defensa
jupyter notebook notebooks/Defensa_Final_Presentacion.ipynb

  Resultados destacados

Clasificación (Random Forest): ROC-AUC ≈ 0.906

Regresión (Random Forest): R² ≈ 0.636, RMSE ≈ 6.87, MAE ≈ 5.14

Clustering (UMAP + KMeans): Agrupación clara de países según PIB y esperanza de vida

 ¿Qué incluye este repositorio?

Limpieza y preparación de datos

Modelado supervisado: clasificación y regresión

Análisis no supervisado: clustering + reducción dimensional (UMAP)

Comparación de métricas y visualizaciones listas para defensa

Reproducibilidad garantizada con Kedro + DVC

Notebook final integrado para la presentación:

notebooks/Defensa_Final_Presentacion.ipynb

  Reproducibilidad

El pipeline completo puede ejecutarse desde cero con:

kedro run


Todos los outputs están controlados por DVC (dvc.yaml)

Los modelos y métricas se guardan automáticamente en carpetas versionadas

Los datos están organizados ordenadamente en /data según cada etapa del pipeline
