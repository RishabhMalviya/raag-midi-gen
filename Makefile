SHELL := /bin/bash

poetry-env:
	@echo "Checking for existing Poetry virtualenv..."
	-poetry env remove --all >/dev/null 2>&1 || true
	@echo "Creating fresh Poetry environment and installing dependencies from pyproject.toml..."
	export POETRY_VIRTUALENVS_IN_PROJECT=true && poetry install --all-groups
	@echo "Done. Use 'poetry run <command>' to run commands inside the environment."

notebook:
	poetry run jupyter notebook --no-browser --port 8080

mlflow:
	cd ./experiments/logs && poetry run mlflow ui



