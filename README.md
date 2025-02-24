# T2S-run-experiment-openai-ft

This repository contains scripts and resources for running and analyzing **Text-to-SQL (T2S)** experiments using **OpenAI GPT models** with fine-tuning. The main goal is to evaluate how well a fine-tuned GPT-based model can convert natural language questions (NLQs) into valid SQL queries and compare its performance across various metrics.

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Experiments](#running-the-experiments)
  - [1. run_experiment.py](#1-run_experimentpy)
  - [2. run_analysis.py](#2-run_analysispy)
  - [3. run_general_analysis.py](#3-run_general_analysispy)
  - [4. run_split_format_data.py](#4-run_split_format_datapy)
- [Results](#results)
- [License](#license)

---

## Overview
This project focuses on converting **Natural Language Queries (NLQs)** into **SQL queries** using **OpenAI GPT** models that have been **fine-tuned** for improved accuracy. It includes scripts to:

- Run experiments using fine-tuned GPT models.
- Analyze the generated SQL in terms of syntax, semantics, and execution results.
- Produce summary metrics and visualizations of the experiment outcomes.
- Prepare datasets for fine-tuning, splitting them into training and testing sets.

By following the steps in this repository, you can:
1. Perform multiple experiments (with different prompts/models).
2. Analyze each experiment’s results.
3. Compare experiments collectively with a final, general analysis.
4. Format and split data for fine-tuning your own GPT model on Text-to-SQL tasks.

---

## Repository Structure
```
analysis/                   # Analysis helper modules (error analysis, match analysis, etc.)
database/                   # Database files or scripts (if any) used in the experiments
dataset/                    # Contains input data or reference materials
findings/                   # Output files generated by run_experiment.py (per-experiment results)
images/                     # Automatically saved plots and visualizations
results/                    # Output files generated by run_analysis.py (analysis per experiment)
config.json                 # Main configuration file (API keys, paths, etc.)
data_experiments.py         # Dictionary or definitions of experiment parameters
prompts_to_use.py           # Collection of prompt templates
requirements.txt            # Python dependencies
run_experiment.py           # (1) Script to execute each experiment
run_analysis.py             # (2) Script to analyze results of a single experiment
run_general_analysis.py     # (3) Script to analyze/compare all experiment results
run_split_format_data.py    # (4) Script to split/format data for fine-tuning
README.md                   # Project documentation (this file)
```

---

## Installation
1. **Clone the repository**:
   ```sh
   git clone https://github.com/criscanon/t2s-run-experiment-openai-ft.git
   cd t2s-run-experiment-openai-ft
   ```
2. **Create and activate a virtual environment (recommended)**:
   ```sh
   python3 -m venv venv
   source venv/bin/activate   # On macOS/Linux
   # or
   venv\Scripts\activate      # On Windows
   ```
3. **Install the required dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

---

## Configuration
The **`config.json`** file contains important settings, such as:
- **`connection_string`**: The SQLAlchemy connection string to your database (e.g., `sqlite:///database/example.db`).
- **`dataset_excel_path`**: Path to the Excel file containing your main NLQ-SQL pairs.
- **`dataset_excel_path_test`**: Path to a separate Excel file used for testing (if applicable).
- **`output_path`**: Directory to store logs and results.
- **`api_key`**: Your OpenAI API key.
- **`<model_name>`**: The specific GPT-based fine-tuned model name you want to use (e.g., `"my-fine-tuned-model"`).
- **`test_size`**: Proportion of the dataset used as the test set when splitting data.

Example of a minimal `config.json`:
```json
{
  "api_key": "<YOUR_OPENAI_API_KEY>",
  "gpt35": "my-fine-tuned-gpt-3.5",
  "connection_string": "sqlite:///database/example.db",
  "dataset_excel_path": "dataset/nlq_sql_pairs.xlsx",
  "dataset_excel_path_test": "dataset/nlq_sql_pairs_test.xlsx",
  "output_path": "findings/",
  "test_size": 0.2
}
```
Make sure to adjust each field to match your environment and requirements.

---

## Running the Experiments

### 1. run_experiment.py
- **Purpose**: Executes an experiment by:
  - Reading a set of NLQs and their expected SQL queries.
  - Generating SQL from a chosen **fine-tuned GPT model** and comparing it to the expected SQL.
  - Logging all attempts, successes, and errors.
  - Saving the raw experiment results in an Excel file inside the `findings/` folder.

- **How to use**:
  1. Open `data_experiments.py` and define or check the experiments you want to run (their ID, model, prompt, etc.).
  2. In `run_experiment.py`, set the `id_experiment` to the ID of the experiment you wish to run.
  3. Run:
     ```sh
     python run_experiment.py
     ```
  4. After completion, check the `findings/` folder for the generated `*_res.xlsx` file and log file.

**Important**: Repeat this step **for each experiment** you want to conduct.

### 2. run_analysis.py
- **Purpose**: Analyzes the results of a single experiment by:
  - Comparing the expected SQL with the generated SQL.
  - Classifying errors (syntactic, semantic, etc.).
  - Calculating match rates for queries, rows, and columns.
  - Producing an Excel file with detailed analysis in the `results/` folder.

- **How to use**:
  1. Set the same `id_experiment` used in the previous step (or the one you want to analyze) inside `run_analysis.py`.
  2. Run:
     ```sh
     python run_analysis.py
     ```
  3. Check the `results/` folder for the `*_analysis.xlsx` file.

**Important**: Repeat this step **for each experiment** whose results you want to analyze.

### 3. run_general_analysis.py
- **Purpose**: Consolidates and compares the analysis from multiple experiments. Generates plots and metrics to help you visualize and compare different experiments side-by-side.

- **How to use**:
  1. Modify the parameters inside `run_general_analysis.py` (e.g., which models or experiment names to load).
  2. Run:
     ```sh
     python run_general_analysis.py
     ```
  3. Visualizations and comparison results will be saved into the `images/` folder. This script also reads from the `results/` folder to gather each experiment’s analysis summary.

### 4. run_split_format_data.py
- **Purpose**: Splits your dataset into training and test sets, then converts them into **JSONL** format suitable for fine-tuning with OpenAI’s API. Also saves separate Excel files for training and test data.
- **How to use**:
  1. In `run_split_format_data.py`, set the `id_experiment` and verify the input file (`dataset_excel_path`) and test size in `config.json`.
  2. Run:
     ```sh
     python run_split_format_data.py
     ```
  3. Check the `dataset/` folder for the generated training and test files (both `.jsonl` and `.xlsx`).

---

## Results
- **`findings/`**: Contains raw experiment outputs (e.g., `experiment_name_res.xlsx`) for each run.
- **`results/`**: Contains analyzed data (e.g., `experiment_name_analysis.xlsx`) for each experiment.
- **`images/`**: Stores charts and plots produced by the general analysis script.

Use these outputs to:
- Compare how well your fine-tuned GPT model performed.
- Check error classifications and reasons for SQL generation failures.
- Evaluate metrics like match rate (SQL syntax, returned rows/columns, etc.) and execution time.

---

## License
This project is released under the [MIT License](LICENSE). Feel free to use and adapt the code according to its terms.

---
