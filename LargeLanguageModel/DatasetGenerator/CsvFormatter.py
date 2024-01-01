import json
import pandas as pd
import csv

def convert_json_to_csv(instructions_json, annotated_ids_json, csv_file_name):
    # Reading JSON data from a file
    with open(json_file_name) as f:
        json_data = json.load(f)

    # Converting JSON data to a pandas DataFrame
    df = pd.DataFrame(json_data)

    # Writing DataFrame to a CSV file
    df.to_csv(csv_file_name, index=False)

def convert_csv_to_json(csv_file_name, json_file_name):
    # Create a dictionary
    data = {}



if __name__ == '__main__':
    json_file_name = 'DatasetFiles/instructions_types.json'
    csv_file_name = 'DatasetFiles/instructions_types_csv.csv'
    convert_json_to_csv(json_file_name, csv_file_name)
