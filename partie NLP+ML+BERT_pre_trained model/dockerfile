# Utilisez une image Python officielle
FROM python:3.8-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY api.py .
COPY requirements.txt .
COPY dataset/intents1.json ./dataset/
COPY model/chatbot_model.pkl ./model/
COPY model/vectorizer.pkl ./model/
COPY templates/index.html ./templates/

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger les modèles NLP
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Exposer le port 5000
EXPOSE 5000

# Commande de démarrage
CMD ["python", "api.py", "--host=0.0.0.0"]