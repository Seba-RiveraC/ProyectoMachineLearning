# Imagen base de Python
FROM python:3.11-slim

# Configuraciones esenciales
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PROJECT_HOME=/opt/proyectomachineml

# Instalación de dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio del proyecto
WORKDIR ${PROJECT_HOME}

# Copiar requirements primero (mejora la velocidad de build)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Instalar DVC dentro del contenedor
RUN pip install --no-cache-dir dvc[s3]

# Copiar TODO el proyecto
COPY . .

# Recuperar los datos versionados (no falla si no hay remoto configurado)
RUN dvc pull || echo "No remote configured, skipping"

# Exponer puerto opcional para Kedro-Viz
EXPOSE 4141

# Comando por defecto — ejecuta todo el pipeline con Kedro
CMD ["kedro", "run"]
