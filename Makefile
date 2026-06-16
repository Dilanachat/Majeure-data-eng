# ==============================================================================
# Projet de classification - Makefile (squelette)
# ==============================================================================
# Seuls les targets d'INSTALLATION sont fournis. Les autres sont a completer
# au fil des TP (un `# TODO (Sx)` indique la commande attendue).
# Environnement gere par uv (Python 3.13) a partir de pyproject.toml.
# Aide : make help
# ==============================================================================

SHELL        := /bin/sh
# Si le venv est déjà activé (VIRTUAL_ENV défini), on utilise python directement
ifdef VIRTUAL_ENV
PYTHON       := python
RUN          :=
else
PYTHON       := uv run python
RUN          := uv run
endif
VENV_DIR     := .venv
PYTHONPATH   ?= .
export PYTHONPATH
API_HOST     ?= 127.0.0.1
API_PORT     ?= 8000
FRONTEND_PORT ?= 8501
MLFLOW_PORT  := 5000
C            ?= 1.0
MAX_ITER     ?= 1000
CV           ?= 5
SCORING      ?= roc_auc
N_TRIALS     ?= 30

# Couleurs ANSI
YELLOW := $(shell printf '\033[33m')
GREEN  := $(shell printf '\033[32m')
RED    := $(shell printf '\033[31m')
CYAN   := $(shell printf '\033[36m')
RESET  := $(shell printf '\033[0m')

.DEFAULT_GOAL := help

.PHONY: help \
        check-uv check-venv venv-create install sync deps-sync lock reset-env doctor \
        data train train-models train-optuna evaluate \
        mlflow mlflow-local mlflow-run mlflow-down \
        api frontend \
        docker-build docker-run docker-up docker-down \
        lint format type test check


# ==============================================================================
# Help
# ==============================================================================

help: ## Liste des commandes disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(CYAN)%-16s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)


# ==============================================================================
# Setup - Installation de l'environnement Python (uv + pyproject.toml) [FOURNI]
# ==============================================================================

check-uv:
	@command -v uv >/dev/null 2>&1 || { \
		echo "$(RED)[ERREUR] uv n'est pas installe$(RESET)"; \
		echo "  Installation : https://docs.astral.sh/uv/"; \
		exit 1; \
	}

check-venv:
	@test -d $(VENV_DIR) || { \
		echo "$(RED)[ERREUR] Virtualenv manquant : $(VENV_DIR)$(RESET)"; \
		echo "  Lance : make install"; \
		exit 1; \
	}

venv-create: check-uv ## Cree un virtualenv vide (.venv)
	@echo "$(YELLOW)>> Creation du virtualenv...$(RESET)"
	uv venv $(VENV_DIR)
	@echo "$(GREEN)[OK] Virtualenv cree$(RESET)"

deps-sync: check-uv ## Synchronise les dependances projet + dev (uv sync)
	@echo "$(YELLOW)>> Synchronisation des dependances...$(RESET)"
	uv sync --extra dev
	@echo "$(GREEN)[OK] Dependances installees$(RESET)"

install: deps-sync ## Cree le venv et installe le projet + dev (alias)

sync: deps-sync ## Alias de deps-sync

lock: check-uv ## Genere/actualise uv.lock depuis pyproject.toml
	@echo "$(YELLOW)>> Generation du lockfile...$(RESET)"
	uv lock
	@echo "$(GREEN)[OK] uv.lock genere$(RESET)"

reset-env: check-uv ## Reinitialise l'environnement (.venv + uv.lock)
	@echo "$(YELLOW)>> Reinitialisation de l'environnement...$(RESET)"
	rm -rf $(VENV_DIR) uv.lock
	uv sync --extra dev
	@echo "$(GREEN)[OK] Environnement recree$(RESET)"

doctor: check-uv check-venv ## Diagnostique l'environnement de travail
	@uv --version
	@$(PYTHON) --version
	@echo "$(GREEN)[OK] Environnement pret$(RESET)"


# ==============================================================================
# Pipeline ML
# ==============================================================================

data: check-venv ## Echantillonne les donnees (20 000 lignes) et construit les features -> data/train_features.csv
	@echo "$(YELLOW)>> Preparation des donnees...$(RESET)"
	$(PYTHON) -c "\
from src.data import load_data, sample_stratified; \
from src.feature import build_features; \
df = build_features(sample_stratified(load_data())); \
df.to_csv('data/train_features.csv', index=False); \
print(f'Sauvegarde : data/train_features.csv  ({len(df):,} lignes x {df.shape[1]} colonnes)')"
	@echo "$(GREEN)[OK] Donnees prete$(RESET)"

train: check-venv ## Entraine le modele choisi avec GridSearchCV + log MLflow (MODEL=rf|xgb|lgbm|all)
	@echo "$(YELLOW)>> Entrainement du modele : $(MODEL)...$(RESET)"
	$(PYTHON) -m src.train --model $(MODEL)
	@echo "$(GREEN)[OK] Entrainement termine$(RESET)"

train-models: check-venv ## Entraine et compare RF + XGBoost + LightGBM -> MLflow
	@echo "$(YELLOW)>> Comparaison des 3 modeles...$(RESET)"
	$(PYTHON) -m src.train --model all
	@echo "$(GREEN)[OK] Voir resultats sur http://127.0.0.1:$(MLFLOW_PORT)$(RESET)"

train-optuna: check-venv ## Optimise les hyperparametres avec Optuna (MODEL=rf|xgb|lgbm|all N_TRIALS=30)
	@echo "$(YELLOW)>> Optimisation Optuna : $(MODEL) — $(N_TRIALS) trials...$(RESET)"
	$(PYTHON) -m src.train --model $(MODEL) --optimizer optuna --n-trials $(N_TRIALS)
	@echo "$(GREEN)[OK] Voir resultats sur http://127.0.0.1:$(MLFLOW_PORT)$(RESET)"

evaluate: check-venv ## Evalue + valide la derniere version du modele (mlflow.evaluate + seuils)
	@echo "$(YELLOW)>> Evaluation du dernier modele...$(RESET)"
	$(PYTHON) -m src.evaluate
	@echo "$(GREEN)[OK] Evaluation terminee$(RESET)"

mlflow: check-venv ## Demarre le serveur MLflow local (sqlite) sur le port 5000
	@echo "$(YELLOW)>> Demarrage MLflow sur http://127.0.0.1:$(MLFLOW_PORT)$(RESET)"
	$(RUN) mlflow server \
		--host 127.0.0.1 \
		--port $(MLFLOW_PORT) \
		--backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlruns

mlflow-local: check-venv ## Demarre MLflow sans docker (sqlite + serve-artifacts) -> http://127.0.0.1:5000
	@echo "$(YELLOW)>> Demarrage MLflow local sur http://$(API_HOST):$(MLFLOW_PORT)$(RESET)"
	$(PYTHON) -m mlflow server \
		--backend-store-uri sqlite:///mlflow.db \
		--artifacts-destination ./mlartifacts --serve-artifacts \
		--host $(API_HOST) --port $(MLFLOW_PORT)

mlflow-run: check-venv ## Lance un entry point MLproject dans le venv courant (ENTRY=train PARAMS=)
	MLFLOW_TRACKING_URI="$$($(PYTHON) -c 'from src.config import MLFLOW_TRACKING_URI; print(MLFLOW_TRACKING_URI)')" \
	$(PYTHON) -m mlflow run . --env-manager local \
		--experiment-name "$$($(PYTHON) -c 'from src.config import MLFLOW_EXPERIMENT; print(MLFLOW_EXPERIMENT)')" \
		-e $(ENTRY) $(PARAMS)

mlflow-down: ## Arrete le serveur MLflow (docker compose)
	docker compose down

api: ## Lance l'API FastAPI en rechargement auto (voir API_HOST/API_PORT)
	# TODO (S12) : $(RUN) uvicorn src.api:app --reload --host $(API_HOST) --port $(API_PORT)

frontend: ## Lance le frontend Streamlit (voir FRONTEND_PORT)
	# TODO (S14) : $(RUN) streamlit run frontend/app.py --server.port $(FRONTEND_PORT)


# ==============================================================================
# Docker  [A COMPLETER]
# ==============================================================================

docker-build: ## Construit l'image d'entrainement
	# TODO (S8) : docker build -f docker/Dockerfile.train -t mlproject-train .

docker-run: ## Lance l'entrainement en conteneur
	# TODO (S8) : docker run --rm -v "$(CURDIR)/../models:/app/models" mlproject-train

docker-up: ## Demarre la stack (mlflow, api, frontend)
	# TODO (S14) : docker compose -f docker-compose.yml up -d --build mlflow api frontend

docker-down: ## Arrete et supprime les conteneurs (conserve les volumes)
	# TODO (S14) : docker compose -f docker-compose.yml down


# ==============================================================================
# Qualite  [A COMPLETER]
# ==============================================================================

lint: check-venv ## Verifie le style (ruff)
	$(RUN) ruff check src/

format: check-venv ## Formate le code (ruff)
	$(RUN) ruff format src/

type: check-venv ## Verifie les types (mypy)
	$(RUN) mypy src/

test: check-venv ## Lance les tests (pytest)
	$(RUN) pytest

check: lint type test ## Workflow qualite complet (lint + types + tests)
