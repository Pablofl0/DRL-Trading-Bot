# Imagen base con soporte GPU
FROM tensorflow/tensorflow:2.15.0-gpu

# Configuración básica
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Copiar requirements e instalar dependencias
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Instalar utilidades opcionales
RUN apt-get update && apt-get install -y git && apt-get clean

# Copiar todo el proyecto
COPY . /app

# Crear carpetas por si no existen
RUN mkdir -p /app/data /app/logs /app/checkpoints

# Exponer puerto para TensorBoard
EXPOSE 6006

# Comando por defecto (puede cambiarse en docker-compose)
CMD ["python", "train.py"]
