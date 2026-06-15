# Utiliser une image Python officielle légère en tant que base
FROM python:3.12-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Empêcher Python de mettre en cache la sortie (pour voir les print en temps réel)
ENV PYTHONUNBUFFERED=1

# Installer la librairie PostgreSQL pour Python + Les librairies mathématiques
RUN pip install psycopg2-binary python-dotenv poke-env numpy scipy pandas scikit-learn joblib

# Copier votre script dans le conteneur
COPY init_neondb.py .
COPY bot.py .
COPY evolutionary_bot.py .
COPY fitness_engine.py .
COPY nash_solver.py .
COPY simulator_5v5.py .
COPY ai_5v5_predictor.py .
COPY ai_1v1_predictor.py .

# Commande par défaut pour exécuter le script
CMD ["python", "init_neondb.py"]