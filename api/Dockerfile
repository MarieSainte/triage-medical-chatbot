# Utilisation d'une image Python légère
FROM python:3.12-slim

# Installation des dépendances système nécessaires pour PostgreSQL (psycopg2)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Définition du répertoire de travail
WORKDIR /app

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du reste du code de l'API
COPY . .

# Exposition du port utilisé par FastAPI
EXPOSE 8080

# Commande pour lancer l'API avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]