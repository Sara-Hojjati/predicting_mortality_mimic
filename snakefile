# Define the rule for querying and fetching data from the database
rule fetch_data:
    output:
        "data/raw_data.csv"
    script:
        "scripts/fetch_data.py"

# Define the rule for performing EDA on the the raw data
rule eda:
    input:
        "data/raw_data.csv"
    output:
        "results/eda_report.ipynb"
    script:
        "scripts/perform_eda.py"

# Define the rule for preprocessing the fetched data
rule preprocess_data:
    input:
        "data/raw_data.csv"
    output:
        "data/X_train_imputed.csv"
        "data/y_train.csv"
        "data/X_test_imputed.csv"
        "data/y_test.csv"
    script:
        "scripts/preprocess_data.py"

# Define the rule for training and evaluating machine learning models
rule ml_modeling:
    input:
        "data/X_train_imputed.csv"
        "data/y_train.csv"
        "data/X_test_imputed.csv"
        "data/y_test.csv"
    output:
        "results/model_performance.json"
    script:
        "scripts/train_evaluate_models.py"
