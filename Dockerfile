# Dockerfile.api
FROM python:3.12-slim

WORKDIR /app

# Installation des dépendances système (si nécessaire pour psycopg2 par exemple)
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

# Copie et installation des requirements
COPY requirements.txt .
# Astuce : Pour la prod, idéalement, utilisez numpy/torch version CPU pour réduire la taille
RUN pip install --no-cache-dir -r requirements.txt

# --- COPIE SÉLECTIVE (Sécurité : on ne copie pas les scripts d'entrainement) ---
COPY config.py .
COPY api/ ./api/
COPY core/ ./core/
COPY entities/ ./entities/
COPY model/ ./model/

# Copie uniquement les définitions de modèles et les poids (.pt), PAS les scripts train_*.py
#COPY model/gnn_fraud_detector.pt model/content_moderation_model.py ./model/
# Copie de tous les fichiers .pt présents dans le dossier model local
#COPY model/*.pt ./model/

# --- CONFIGURATION ---
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Important : On écrase le chemin Windows du .env pour un chemin Linux
#ENV MODEL_PATH=/app/model/gnn_fraud_detector.pt

# Port exposé
EXPOSE 8000

# Lancement de l'API (ajustez 'api.inference:app' selon où se trouve votre objet FastAPI)
CMD ["uvicorn", "api.inference:app", "--host", "0.0.0.0", "--port", "8000"]
