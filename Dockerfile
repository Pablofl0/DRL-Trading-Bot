# Usa imagen base de TensorFlow que ya incluye soporte GPU y CPU
# TensorFlow detectará automáticamente si hay GPU disponible
FROM tensorflow/tensorflow:2.15.0

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    python3-dev \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Define directorio de trabajo
WORKDIR /app

# Copia archivos del proyecto
COPY . .

# Actualiza pip, setuptools y wheel usando python3 -m pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copia el requirements.txt y lo instala usando python3 -m pip
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Forzar TensorFlow a usar solo CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=2


# Comando por defecto — se puede sobreescribir desde docker-compose
CMD ["python3", "auto_train.py"]
