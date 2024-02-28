import config
from datasets import Value, load_dataset

def format_instruction_types(dataset):
    features = dataset.features.copy()
    features['instruction_type'] = Value(dtype="string", id=None)
    dataset = dataset.cast(features)

    def modify_instruction_type_field(row):
        if row["instruction_type"] == '0':
            row["instruction_type"] = "ADD"
        elif row["instruction_type"] == '1':
            row["instruction_type"] = "DELETE"
        elif row["instruction_type"] == '2':
            row["instruction_type"] = "EDIT"
        elif row["instruction_type"] == '3':
            row["instruction_type"] = "MOVE"
        return row

    formatted_dataset = dataset.map(modify_instruction_type_field)
    return formatted_dataset

def upload_instruction_type_datasets():
    dataset = load_dataset("McSpicyWithMilo/infographic-instructions",
                           data_files='instructions_400.json',
                           split="train")
    dataset = dataset.remove_columns(["edit_type", "infographic_section", "target_element", "target_location"])
    stratify_column_name = 'instruction_type'
    dataset = dataset.class_encode_column(stratify_column_name).train_test_split(test_size=0.3,
                                                                                 stratify_by_column=stratify_column_name)
    train_dataset = dataset["train"]
    train_dataset = format_instruction_types(train_dataset)
    test_dataset = dataset["test"]
    test_dataset = format_instruction_types(test_dataset)
    train_dataset.push_to_hub("McSpicyWithMilo/instruction-types-0.3split", split="train", token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)
    test_dataset.push_to_hub("McSpicyWithMilo/instruction-types-0.3split", split="test", token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

def format_infographic_sections(dataset):
    features = dataset.features.copy()
    features['infographic_section'] = Value(dtype="string", id=None)
    dataset = dataset.cast(features)

    def modify_infographic_section_field(row):
        if row["infographic_section"] == '0':
            row["infographic_section"] = "Header"
        elif row["infographic_section"] == '1':
            row["infographic_section"] = "Knowledge Graph Summaries"
        elif row["infographic_section"] == '2':
            row["infographic_section"] = "Latest Comments"
        elif row["infographic_section"] == '3':
            row["infographic_section"] = "Number of Shares"
        elif row["infographic_section"] == '4':
            row["infographic_section"] = "Related Facts"
        elif row["infographic_section"] == '5':
            row["infographic_section"] = "Similar Articles"
        elif row["infographic_section"] == '6':
            row["infographic_section"] = "Vote on Reliability"
        return row

    formatted_dataset = dataset.map(modify_infographic_section_field)
    return formatted_dataset

def upload_infographic_section_datasets():
    dataset = load_dataset("McSpicyWithMilo/infographic-instructions",
                           data_files='instructions_400.json',
                           split="train")
    dataset = dataset.remove_columns(["edit_type", "target_element", "target_location"])
    stratify_column_name = 'infographic_section'
    dataset = dataset.class_encode_column(stratify_column_name).train_test_split(test_size=0.3,
                                                                                 stratify_by_column=stratify_column_name)
    train_dataset = dataset["train"]
    train_dataset = format_infographic_sections(train_dataset)
    test_dataset = dataset["test"]
    test_dataset = format_infographic_sections(test_dataset)
    train_dataset.push_to_hub("McSpicyWithMilo/infographic-sections-0.3split", split="train",
                              token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)
    test_dataset.push_to_hub("McSpicyWithMilo/infographic-sections-0.3split", split="test",
                             token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

def upload_target_element_datasets():
    dataset = load_dataset("McSpicyWithMilo/infographic-instructions",
                           data_files='instructions_400.json',
                           split="train")
    dataset = dataset.remove_columns(["edit_type", "infographic_section", "target_location"])
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset.push_to_hub("McSpicyWithMilo/target-elements-0.2split", split="train",
                              token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)
    test_dataset.push_to_hub("McSpicyWithMilo/target-elements-0.2split", split="test",
                             token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

def upload_target_location_datasets():
    dataset = load_dataset("McSpicyWithMilo/infographic-instructions",
                           data_files='instructions_400.json',
                           split="train")
    dataset = dataset.remove_columns(["infographic_section", "target_element", "edit_type"])
    dataset = dataset.filter(lambda example: example["instruction_type"] == 'ADD')
    dataset = dataset.train_test_split(test_size=0.3)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset.push_to_hub("McSpicyWithMilo/target-locations-0.3split", split="train",
                              token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)
    test_dataset.push_to_hub("McSpicyWithMilo/target-locations-0.3split", split="test",
                             token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

def upload_target_element_datasets_new():
    dataset = load_dataset("McSpicyWithMilo/infographic-instructions",
                           data_files='instructions_new.json',
                           split="train")
    dataset = dataset.remove_columns(["infographic_section", "instruction_type", "target_location"])
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset.push_to_hub("McSpicyWithMilo/target-elements-0.2split-new-180", split="train",
                              token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)
    test_dataset.push_to_hub("McSpicyWithMilo/target-elements-0.2split-new-180", split="test",
                             token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

def upload_infographic_section_datasets_new():
    dataset = load_dataset("McSpicyWithMilo/infographic-instructions",
                           data_files='instructions_new.json',
                           split="train")
    dataset = dataset.remove_columns(["target_element", "instruction_type", "target_location"])
    stratify_column_name = 'infographic_section'
    dataset = dataset.class_encode_column(stratify_column_name).train_test_split(test_size=0.2,
                                                                                 stratify_by_column=stratify_column_name)
    train_dataset = dataset["train"]
    train_dataset = format_infographic_sections(train_dataset)
    test_dataset = dataset["test"]
    test_dataset = format_infographic_sections(test_dataset)
    train_dataset.push_to_hub("McSpicyWithMilo/infographic-sections-0.2split-new-180", split="train",
                              token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)
    test_dataset.push_to_hub("McSpicyWithMilo/infographic-sections-0.2split-new-180", split="test",
                             token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

def upload_target_location_datasets_new():
    dataset = load_dataset("McSpicyWithMilo/infographic-instructions",
                           data_files='instructions_new.json',
                           split="train")
    dataset = dataset.remove_columns(["target_element", "instruction_type", "infographic_section"])
    dataset = dataset.train_test_split(test_size=0.2)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    train_dataset.push_to_hub("McSpicyWithMilo/target-locations-0.2split-new-180", split="train",
                              token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)
    test_dataset.push_to_hub("McSpicyWithMilo/target-locations-0.2split-new-180", split="test",
                             token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

def upload_target_location_infographic_section_datasets_new():
    dataset = load_dataset("McSpicyWithMilo/infographic-instructions",
                           data_files='instructions_new.json',
                           split="train")
    dataset = dataset.remove_columns(["instruction", "instruction_type", "target_element"])
    stratify_column_name = 'infographic_section'
    dataset = dataset.class_encode_column(stratify_column_name).train_test_split(test_size=0.2,
                                                                                 stratify_by_column=stratify_column_name)
    train_dataset = dataset["train"]
    train_dataset = format_infographic_sections(train_dataset)
    test_dataset = dataset["test"]
    test_dataset = format_infographic_sections(test_dataset)
    train_dataset.push_to_hub("McSpicyWithMilo/target-location-infographic-section-0.2split-new-180", split="train",
                              token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)
    test_dataset.push_to_hub("McSpicyWithMilo/target-location-infographic-section-0.2split-new-180", split="test",
                             token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

if __name__ == '__main__':
    # upload_instruction_type_datasets()
    # upload_infographic_section_datasets()
    # upload_target_element_datasets()
    # upload_target_location_datasets()
    upload_target_element_datasets_new()
    upload_infographic_section_datasets_new()
    upload_target_location_datasets_new()
    upload_target_location_infographic_section_datasets_new()
