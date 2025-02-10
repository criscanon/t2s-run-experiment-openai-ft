import pandas as pd
import json
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.schema import CreateTable
from prompts_to_use import prompts
from experiments import experiments

# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

def get_db_info():
    """
    Connects to the database using SQLAlchemy and retrieves the dialect and schema information.

    Returns:
    tuple: A tuple containing the dialect name and the schema of the database.
    """
    connection_string = config['connection_string']

    # Connect to the database using SQLAlchemy
    engine = create_engine(connection_string)
    metadata = MetaData()
    metadata.reflect(engine)
    dialect = engine.dialect.name

    # Get the construction statements for each table
    schema_statements = []
    for table_name in metadata.tables:
        table = Table(table_name, metadata, autoload_with=engine)
        ddl = CreateTable(table).compile(engine)
        schema_statements.append(str(ddl))

    # Join all construction statements into a single string
    schema = '\n\n'.join(schema_statements)
    
    # Ensure there are only two newlines between each statement
    schema = '\n'.join(filter(None, schema.split('\n\n')))

    return dialect, schema

def split_and_convert_to_jsonl_and_excel(input_file, output_file_train_jsonl, output_file_test_jsonl, output_file_train_excel, output_file_test_excel, test_size):
    """
    Reads an Excel file, formats the 'nlq' column with the prompt, splits the data into
    training and test sets according to the specified proportion, and writes these sets
    to new JSONL and Excel files.

    Parameters:
    input_file (str): The path to the input Excel file.
    output_file_train_jsonl (str): The path to the JSONL file where the training data will be saved.
    output_file_test_jsonl (str): The path to the JSONL file where the test data will be saved.
    output_file_train_excel (str): The path to the Excel file where the training data will be saved.
    output_file_test_excel (str): The path to the Excel file where the test data will be saved.
    test_size (float): The proportion of the dataset to include in the test split (between 0.0 and 1.0).
    """
    # Read the Excel file
    df = pd.read_excel(input_file)

    # Get database information
    dialect, schema = get_db_info()

    # Prepare the prompt
    prompt_to_use = prompts[experiments[id_experiment]["prompt"]]
    prompt_completed = prompt_to_use.format(dialect=dialect, schema=schema, nlq="{nlq}")

    # Format the 'nlq' column directly
    if 'nlq' in df.columns:
        df['nlq'] = df['nlq'].apply(lambda nlq: prompt_completed.format(nlq=nlq))

    # Calculate the split index
    total_rows = len(df)
    test_size_rows = int(total_rows * test_size)

    # Shuffle the DataFrame rows
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the DataFrame
    df_test = df_shuffled.iloc[:test_size_rows]
    df_train = df_shuffled.iloc[test_size_rows:]

    # Function to write to JSONL file
    def write_to_jsonl(df, output_file):
        with open(output_file, 'w') as f:
            for index, row in df.iterrows():
                # Build the user message
                user_message = {"role": "user", "content": row['nlq']}
                
                # Build the assistant message
                assistant_message = {"role": "assistant", "content": row['sql']}
                
                # Build the system message
                system_message = {"role": "system", "content": "T2S is a Text to SQL converter"}
                
                # Write each message as a line in the JSONL file
                f.write(json.dumps({"messages": [system_message, user_message, assistant_message]}) + '\n')

    # Save the training and test data to separate JSONL files
    write_to_jsonl(df_train, output_file_train_jsonl)
    write_to_jsonl(df_test, output_file_test_jsonl)

    # Save the training and test data to separate Excel files
    df_train.to_excel(output_file_train_excel, index=False)
    df_test.to_excel(output_file_test_excel, index=False)

    print(f"Training file saved as {output_file_train_jsonl} and {output_file_train_excel}")
    print(f"Test file saved as {output_file_test_jsonl} and {output_file_test_excel}")

# Specify the input and output file names and the test size proportion
id_experiment = "9-1"
input_file = config['dataset_excel_path']
output_file_train_jsonl = "dataset/" + experiments[id_experiment]["name"] + "_train.jsonl"
output_file_test_jsonl = "dataset/" + experiments[id_experiment]["name"]+ "_test.jsonl"
output_file_train_excel = "dataset/" + experiments[id_experiment]["name"] + "_train.xlsx"
output_file_test_excel = "dataset/" + experiments[id_experiment]["name"]+ "_test.xlsx"
test_size = config['test_size']  # Proportion of the dataset to be used as the test set (between 0.0 and 1.0)

# Call the function to split and save the file
split_and_convert_to_jsonl_and_excel(input_file, output_file_train_jsonl, output_file_test_jsonl, output_file_train_excel, output_file_test_excel, test_size)
