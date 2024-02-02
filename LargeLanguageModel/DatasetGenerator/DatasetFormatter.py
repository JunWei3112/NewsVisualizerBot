import json

def load_annotations(annotations_json_file_name):
    file_obj = open(annotations_json_file_name, "r")
    json_content = file_obj.read()
    return json.loads(json_content)

def save_dataset(dataset_json_file_name, dataset):
    json_string_dataset = json.dumps(dataset)
    json_file_dataset = open(dataset_json_file_name, "w")
    json_file_dataset.write(json_string_dataset)
    json_file_dataset.close()

def extract_instruction_types():
    annotations_json_file_name = 'bard_data_files/annotations.json'
    annotations = load_annotations(annotations_json_file_name)

    instructions_and_types = list()

    for annotation in annotations:
        instruction_and_type_json = {
            "instruction": annotation["instruction"]["prompt"],
            "instruction_type": annotation["annotation"]["instruction_type"]
        }
        instructions_and_types.append(instruction_and_type_json)

    dataset_json_file_name = 'bard_data_files/instructions_types.json'
    save_dataset(dataset_json_file_name, instructions_and_types)

if __name__ == '__main__':
    extract_instruction_types()


