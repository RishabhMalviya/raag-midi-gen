Run `source setup.sh` to set things up.

# Usage Instructions
1. Run whatever experiment you want with a command like `python ./experiments/scripts/<experiment-name>/<main-script>`. If you've built off of the base examples in the cookiecutter project, it should save everything to MLFlow logs under the `experiments/logs/mlruns` directory.
2. To access the MLFlow UI, preferably run `mlflow ui` from the the `experiments/logs` directory. Running the command from other directories will pollute your file structure with stray `mlruns` directories.

# Directory Structure
1. `data` is meant hold the actual data and some notebooks for exploration
2. `eda` is meant to hold notebooks and generated plots from EDA
3. `experiments` is meant to hold training/evaluation scripts and notebooks.
4. The folder with the name of the project is mean to hold python modules that you want to be able to import everywhere else.
