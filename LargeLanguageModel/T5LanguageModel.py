from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers
import torch
from linetimer import CodeTimer
import json

def read_json_file(json_file_name):
    file_obj = open(json_file_name, "r")
    json_content = file_obj.read()
    return json.loads(json_content)

def run_diagnostics(llm_generate_text):
    annotations_json = read_json_file("annotations.json")
    number_of_annotations = 0
    correct_annotations = 0

    with CodeTimer('Identify Instruction Type', unit='s'):
        for annotation in annotations_json:
            instruction = annotation["instruction"]["prompt"]
            expected_instruction_type = annotation["annotation"]["instruction_type"]
            number_of_annotations += 1

            prompt_instruction_type = f'For the following instruction to modify an infographic ({instruction}), what is the instruction type (ADD/DELETE/EDIT/MOVE)?'
            instruction_type_obj = llm_generate_text(prompt_instruction_type)
            instruction_type = instruction_type_obj[0]['generated_text']
            print(f'Instruction: {instruction}')
            print(f'Instruction Type: {instruction_type}')
            print(f'Expected Instruction Type: {expected_instruction_type}')
            print('------------------------------')

            if instruction_type == expected_instruction_type:
                correct_annotations += 1

    print('-----------STATS--------------')
    print(f'Number of correct annotations: {correct_annotations}/{number_of_annotations}')
    print('------------------------------')

def generate_pipeline(model_path):
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
    prompt_instruction_type = f'For the following instruction to modify an infographic ({instruction}), what is the instruction type (ADD/DELETE/EDIT/MOVE)?'

    instruction_type_obj = pipeline(prompt_instruction_type)
    instruction_type = instruction_type_obj[0]['generated_text']
    return instruction_type

def identify_target_element(pipeline, instruction, instruction_type):
    prompt_target_element = f'For the following instruction to modify an infographic ({instruction}), what is the element that is to be '
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
    if instruction_type == 'ADD':
        prompt_infographic_section = f'For the following instruction to modify an infographic ({instruction}), what is the infographic section (1) Number of Shares, 2) Vote on Reliability, 3) Related Facts, 4) Latest Comments, 5) Knowledge Graph Summaries, 6) Similar Articles, 7) Header) where the target element will be added to? '
    elif instruction_type == 'DELETE':
        prompt_infographic_section = f'For the following instruction to modify an infographic ({instruction}), what is the infographic section (1) Number of Shares, 2) Vote on Reliability, 3) Related Facts, 4) Latest Comments, 5) Knowledge Graph Summaries, 6) Similar Articles, 7) Header) where the target element will be deleted from? '
    elif instruction_type == 'EDIT':
        prompt_infographic_section = f'For the following instruction to modify an infographic ({instruction}), what is the infographic section (1) Number of Shares, 2) Vote on Reliability, 3) Related Facts, 4) Latest Comments, 5) Knowledge Graph Summaries, 6) Similar Articles, 7) Header) where the modified element is in? '
    else:
        prompt_infographic_section = f'For the following instruction to modify an infographic ({instruction}), what is the infographic section (1) Number of Shares, 2) Vote on Reliability, 3) Related Facts, 4) Latest Comments, 5) Knowledge Graph Summaries, 6) Similar Articles, 7) Header) where the target element is originally in?  '
    prompt_infographic_section += 'If unable to infer the infographic section, set the answer as NONE.'

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
    model_path = 'google/flan-t5-large-instruction-type-tuned'
    generate_text = generate_pipeline(model_path)

    instruction = "add a comment: 'This is a good article'"
    print('-------------------------------------')
    print(f'Instruction: {instruction}')
    print('-------------------------------------')

    generate_structured_output(generate_text, instruction)

