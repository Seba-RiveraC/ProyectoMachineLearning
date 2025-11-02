Proyecto ML: Análisis de Correlación entre PIB y Esperanza de Vida

(tuve que resubir el readme por problemas con las branch, compasion por favor)

Powered by Kedro

🧠 Descripción General

Este proyecto analiza la relación entre el Producto Interno Bruto (PIB) y la Esperanza de Vida utilizando Machine Learning y la metodología CRISP-DM, implementada sobre el framework Kedro para garantizar un flujo de datos reproducible, escalable y automatizado.

A través de pipelines estructurados, se procesan y limpian datos económicos y demográficos para construir modelos de regresión y clasificación, capaces de estimar y categorizar la esperanza de vida de los países según su desempeño económico.



🚀 Flujo CRISP-DM Implementado

Business Understanding
Se define la hipótesis: “El PIB per cápita está correlacionado con la esperanza de vida de una población.”

Data Understanding
Se integran tres fuentes:

PIB por país y organización (World Bank)

Esperanza de vida por país (ONU)

Series temporales históricas (años 1960–2020)

Data Preparation

Limpieza y estandarización de columnas.

Unificación por país y año.

Eliminación de nulos y valores extremos.

Modeling

Regresión: Random Forest Regressor, con R² = 0.63.

Clasificación: Random Forest Classifier, con ROC-AUC = 0.90.

Optimización con GridSearchCV para ajuste fino de hiperparámetros.

Evaluation

Métricas exportadas automáticamente (.csv en data/07_model_output/).

Visualización de curvas ROC y gráficos de importancia de variables.

🧩 Tecnologías Utilizadas

Python 3.11+

Kedro 0.19+

Scikit-learn

Pandas / NumPy / Matplotlib / Seaborn

Jupyter Notebook

Git + GitHub

Docker para entorno reproducible, el cual no funciono

📈 Resultados Clave

Se confirma una correlación positiva entre el PIB per cápita y la esperanza de vida.

Los países con mayor riqueza tienden a tener poblaciones más longevas.

Los modelos de bosques aleatorios (Random Forest) fueron los más robustos frente a outliers y distribuciones sesgadas.

💡 Conclusiones

El proyecto demuestra la viabilidad de usar variables económicas como predictores de calidad de vida, aplicando un flujo reproducible y automatizado con Kedro.
La arquitectura creada permite escalar hacia análisis predictivos más complejos (por ejemplo, predicción de esperanza de vida futura o impacto de crecimiento económico).
