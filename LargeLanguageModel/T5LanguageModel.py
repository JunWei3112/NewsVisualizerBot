from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers
import torch
from linetimer import CodeTimer
import json
from datasets import *

def get_remote_dataset_json(repo_name, json_file_name):
    dataset_dict = load_dataset(repo_name, data_files=json_file_name)
    train_dataset = dataset_dict["train"]
    train_data_list = list(train_dataset)
    return json.loads(json.dumps(train_data_list))

def run_local_diagnostics_instruction_type(pipeline, hub_dataset_repo_name, hub_dataset_json_file_name):
    annotations_json = get_remote_dataset_json(hub_dataset_repo_name, hub_dataset_json_file_name)
    number_of_annotations = 0
    correct_annotations = 0

    with CodeTimer('Identify Instruction Type', unit='s'):
        for annotation in annotations_json:
            instruction = annotation["instruction"]
            expected_instruction_type = annotation["instruction_type"]
            number_of_annotations += 1

            instruction_type = identify_instruction_type(pipeline, instruction)
            # print(f'Instruction: {instruction}')
            # print(f'Instruction Type: {instruction_type}')
            # print(f'Expected Instruction Type: {expected_instruction_type}')
            # print('------------------------------')

            if instruction_type == expected_instruction_type:
                correct_annotations += 1

            if number_of_annotations % 10 == 0:
                print('-----------UPDATE--------------')
                print(f'Number of correct annotations: {correct_annotations}/{number_of_annotations}')
                print('------------------------------')

    print('-----------FULL STATS--------------')
    print(f'Number of correct annotations: {correct_annotations}/{number_of_annotations}')
    print('------------------------------')

def run_local_diagnostics_infographic_section(pipeline, hub_dataset_repo_name, hub_dataset_json_file_name):
    annotations_json = get_remote_dataset_json(hub_dataset_repo_name, hub_dataset_json_file_name)
    number_of_annotations = 0
    correct_annotations = 0

    with CodeTimer('Identify Infographic Section', unit='s'):
        for annotation in annotations_json:
            instruction = annotation["instruction"]
            expected_instruction_type = annotation["instruction_type"]
            expected_infographic_section = annotation["infographic_section"]
            number_of_annotations += 1

            infographic_section = identify_infographic_section(pipeline, instruction, expected_instruction_type)
            print(f'Instruction: {instruction}')
            print(f'Infographic Section: {infographic_section}')
            print(f'Expected Infographic Section: {expected_infographic_section}')
            print('------------------------------')

            if infographic_section == expected_infographic_section:
                correct_annotations += 1

            if number_of_annotations % 10 == 0:
                print('-----------UPDATE--------------')
                print(f'Number of correct annotations: {correct_annotations}/{number_of_annotations}')
                print('------------------------------')

    print('-----------FULL STATS--------------')
    print(f'Number of correct annotations: {correct_annotations}/{number_of_annotations}')
    print('------------------------------')

def generate_local_pipeline(model_path):
    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    with CodeTimer('Model Loading', unit='s'):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print(f'Model being used: {model_path}')

    with CodeTimer('Tokenizer Loading', unit='s'):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.model_max_length = 2048

    with CodeTimer('Pipeline Loading', unit='s'):
        generate_text = transformers.pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=256,
        )

    return generate_text

def identify_instruction_type(pipeline, instruction):
    # prompt_instruction_type = f'For the following instruction to modify an infographic ({instruction}), what is the instruction type (ADD/DELETE/EDIT/MOVE)?'
    prompt_instruction_type = f'Given the following instruction to modify an infographic: "{instruction}" Identify the type of operation specified in the instruction: ADD/DELETE/EDIT/MOVE.'

    instruction_type_obj = pipeline(prompt_instruction_type)
    instruction_type = instruction_type_obj[0]['generated_text']
    return instruction_type

def identify_target_element(pipeline, instruction, instruction_type):
    prompt_target_element = f'For the following instruction to modify an infographic ({instruction}), what is the element that is to be '
    # prompt_target_element = f'Given the instruction, ({instruction}) Output the specific content or element that is added in the infographic. Ensure that the answer does not include the target location of the added element or text.'
    if instruction_type == 'ADD':
        prompt_target_element += 'added?'
    elif instruction_type == 'DELETE':
        prompt_target_element += 'deleted?'
    elif instruction_type == 'EDIT':
        prompt_target_element += 'modified?'
    else:
        prompt_target_element += 'moved?'

    target_element_obj = pipeline(prompt_target_element)
    target_element = target_element_obj[0]['generated_text']
    return target_element

def identify_infographic_section(pipeline, instruction, instruction_type):
    # if instruction_type == 'ADD':
    #     prompt_infographic_section = f'For the following instruction to modify an infographic ({instruction}), what is the infographic section (Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header) where the target element will be added to? '
    # elif instruction_type == 'DELETE':
    #     prompt_infographic_section = f'For the following instruction to modify an infographic ({instruction}), what is the infographic section (Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header) where the target element will be deleted from? '
    # elif instruction_type == 'EDIT':
    #     prompt_infographic_section = f'For the following instruction to modify an infographic ({instruction}), what is the infographic section (Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header) where the modified element is in? '
    # else:
    #     prompt_infographic_section = f'For the following instruction to modify an infographic ({instruction}), what is the infographic section (Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header) where the target element is originally in?  '
    # prompt_infographic_section += 'If unable to infer the infographic section, set the answer as NONE.'

    infographic_sections = "The infographic comprises a 'Header' section featuring the title and a QR code linking to the content. The 'Number of Shares' section displays the numerical count of shares. The 'Vote on Reliability' section presents a diagram reflecting user opinions on the news article's reliability. The 'Related Facts' section lists statements related to the article, while the 'Latest Comments' section displays user-submitted comments. The 'Knowledge Graph Summaries' section showcases sentiments towards various entities mentioned in the news through a knowledge graph. Lastly, 'Similar Articles' provides a list of articles with diverse viewpoints, each accompanied by a QR code, header, and a brief summary. "
    if instruction_type == 'ADD':
        task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element will be added to within the existing infographic in response to '{instruction}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
    elif instruction_type == 'DELETE':
        task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element will be deleted from within the existing infographic in response to '{instruction}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
    elif instruction_type == 'EDIT':
        task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element will be edited or modified within the existing infographic in response to '{instruction}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
    else:
        task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element is originally in within the existing infographic in response to '{instruction}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
    prompt_infographic_section = infographic_sections + task

    infographic_section_obj = pipeline(prompt_infographic_section)
    infographic_section = infographic_section_obj[0]['generated_text']
    return infographic_section

def identify_target_location(pipeline, instruction, instruction_type):
    if instruction_type == 'ADD':
        prompt_target_location = f'For the following instruction to modify an infographic, ({instruction}), what is the target location of the new element in the infographic section? If unable to infer the location, set the answer as NONE.'
    elif instruction_type == 'MOVE':
        prompt_target_location = f'For the following instruction to modify an infographic, ({instruction}), what is the new location of the target element?'

    target_location_obj = pipeline(prompt_target_location)
    target_location = target_location_obj[0]['generated_text']
    return target_location

def identify_edit_attribute(pipeline, instruction):
    prompt_edit_attribute = f'For the following instruction to modify an infographic, ({instruction}), if the size of the target element is modified, set the answer as SIZE. Else, if the textual content of the target element is modified, set the answer as CONTENT.'

    edit_attribute_obj = pipeline(prompt_edit_attribute)
    edit_attribute = edit_attribute_obj[0]['generated_text']
    return edit_attribute

def identify_edit_value(pipeline, instruction):
    prompt_edit_value = f'For the following instruction to modify an infographic, ({instruction}), what is the new size or textual content of the target element?'

    edit_value_obj = pipeline(prompt_edit_value)
    edit_value = edit_value_obj[0]['generated_text']
    return edit_value

def generate_structured_output(pipeline, instruction):
    with CodeTimer('Generating Structured Output', unit='s'):
        # Identify Instruction Type
        instruction_type = identify_instruction_type(pipeline, instruction)
        print(f'Instruction Type: {instruction_type}')

        # Identify Target Element
        target_element = identify_target_element(pipeline, instruction, instruction_type)
        print(f'Target Element: {target_element}')

        # Identify Infographic Section
        infographic_section = identify_infographic_section(pipeline, instruction, instruction_type)
        print(f'Infographic Section: {infographic_section}')

        # Identify Target Location of Element
        if instruction_type == 'ADD' or instruction_type == 'MOVE':
            target_location = identify_target_location(pipeline, instruction, instruction_type)
            print(f'Target Location: {target_location}')

        # Identify Edit Attribute and Target Value
        if instruction_type == 'EDIT':
            edit_attribute = identify_edit_attribute(pipeline, instruction)
            print(f'Edit Attribute: {edit_attribute}')

            edit_value = identify_edit_value(pipeline, instruction)
            print(f'Edit Value: {edit_value}')

if __name__ == '__main__':
    # model_path = 'google/flan-t5-large-instruction-type-tuned'
    # generate_text = generate_pipeline(model_path)
    #
    # instruction = "add a comment: 'This is a good article'"
    # print('-------------------------------------')
    # print(f'Instruction: {instruction}')
    # print('-------------------------------------')
    #
    # generate_structured_output(generate_text, instruction)

    generate_text_pipeline = generate_local_pipeline('google/flan-t5-large-infographic-section-400-lora')
    hub_dataset_repo_name = "McSpicyWithMilo/infographic-instructions"
    hub_dataset_json_file_name = "instructions_400.json"
    run_local_diagnostics_instruction_type(generate_text_pipeline, hub_dataset_repo_name, hub_dataset_json_file_name)
