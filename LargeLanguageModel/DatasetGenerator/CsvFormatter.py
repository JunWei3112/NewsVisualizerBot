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
    df = pd.read_excel(csv_file_name, usecols=['instruction', 'instruction_type', 'infographic_section', 'edit_type', 'target_element'])

    def modify_infographic_section_field(row):
        if row['infographic_section'] == 1:
            return 'Number of Shares'
        elif row['infographic_section'] == 2:
            return 'Vote on Reliability'
        elif row['infographic_section'] == 3:
            return 'Related Facts'
        elif row['infographic_section'] == 4:
            return 'Latest Comments'
        elif row['infographic_section'] == 5:
            return 'Knowledge Graph Summaries'
        elif row['infographic_section'] == 6:
            return 'Similar Articles'
        elif row['infographic_section'] == 7:
            return 'Header'

    df['infographic_section'] = df.apply(modify_infographic_section_field, axis=1)

    selected_columns = ['instruction', 'instruction_type', 'infographic_section', 'edit_type', 'target_element']
    json_data = df[selected_columns].to_json(orient='records')

    with open(json_file_name, 'w') as json_file:
        json_file.write(json_data)


if __name__ == '__main__':
    json_file_name = 'DatasetFiles/instructions_400.json'
    csv_file_name = 'DatasetFiles/instructions_400.xlsm'
    convert_csv_to_json(csv_file_name, json_file_name)
