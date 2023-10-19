from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers
import torch
from linetimer import CodeTimer

if __name__ == '__main__':
    model_path = "google/flan-t5-large"
    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    with CodeTimer('Model Loading', unit='s'):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print(f'Model being used: {model_path}')

    with CodeTimer('Tokenizer Loading', unit='s'):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
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

    instruction = "Delete the statement 'The news article is based on a scientific study' from the list of statements on related facts."

    prompt_template = f"""
The infographic has the following sections: 1) Number of Shares, 2) Vote on Reliability, 3) Related Facts, 4) Latest Comments,
5) Knowledge Graph Summaries, 6) Similar Articles, 7) Header
From the following instruction ({instruction}), extract the necessary information required to fill up the expected output

If the instruction adds a new element to an infographic, the expected output should be formatted in the following schema:
(
    InstructionType: ADD
    TargetElement: // element that is to be added
    InfographicSection: // try to infer the infographic section that the element will be added to. If unable to infer the section, set the value as NONE
    TargetLocation: // location of the new element in the infographic section, if specified in the instruction. If not specified, set the value as NONE
)
If the instruction modifies an attribute of an existing element in the infographic, the expected output should be formatted in the following schema:
(
    InstructionType: EDIT
    TargetElement: // element that is to be modified
    InfographicSection: // try to infer the infographic section where the modified element is in. If unable to infer the section, set the value as NONE
    ElementLocation: // location of the modified element in the infographic section, if specified in the instruction. If not specified, set the value as NONE
    EditAttribute: // [SIZE] if the size of the element is modified, [CONTENT] if the textual content of the element is modified
    TargetValue: // new size or textual content of the element, if specified in the instruction. If not specified, set the value as NONE
)
If the instruction modifies the location of an existing element in an infographic, the expected output should be formatted in the following schema:
(
    InstructionType: MOVE
    TargetElement: // element that is to be moved
    InfographicSection: // try to infer the infographic section where the target element is originally in. If unable to infer the section, set the value as NONE
    ElementLocation: // original location of the element in the infographic section, if specified in the instruction. If not specified, set the value as NONE
    TargetLocation: // new location of the element, if specified in the instruction. If not specified, set the value as NONE
)
If the instruction removes an existing element from an infographic, the expected output should be formatted in the following schema:
(
    InstructionType: DELETE
    TargetElement: // element that is to be deleted
    InfographicSection: // try to infer the infographic section where the deleted element is in. If unable to infer the section, set the value as NONE
    ElementLocation: // location of the deleted element in the infographic section, if specified in the instruction. If not specified, set the value as NONE
)
    """

    with CodeTimer('Identify Instruction Type', unit='s'):
        prompt_instruction_type = f'''
        For the following instruction to modify an infographic ({instruction}), what is the instruction type?
        If the instruction adds a new element to an infographic, the instruction type is ADD.
        If the instruction modifies an attribute of an existing element in the infographic, the instruction type is EDIT.
        If the instruction modifies the location of an existing element in an infographic, the instruction type is MOVE.
        If the instruction removes an existing element from an infographic, the instruction type is DELETE.
        '''
        instruction_type_obj = generate_text(prompt_instruction_type)
        print(f'Instruction Type: {instruction_type_obj[0]}')

    with CodeTimer('Identify Target Element', unit='s'):
        prompt_target_element = f'''
        For the following instruction to modify an infographic ({instruction}), what is the target element to be added, modified, moved or deleted?
        '''
        target_element_obj = generate_text(prompt_target_element)
        print(f'Target Element: {target_element_obj[0]}')

    with CodeTimer('Identify Infographic Section', unit='s'):
        prompt_infographic_section = f'''
        The infographic has the following sections: 1) Number of Shares, 2) Vote on Reliability, 3) Related Facts, 4) Latest Comments, 5) Knowledge Graph Summaries, 6) Similar Articles, 7) Header
        For the following instruction to modify an infographic ({instruction}), what is the infographic section where the target element to be added, modified, moved or deleted in? If unable to infer the section, set the value as NONE
        '''
        infographic_section_obj = generate_text(prompt_infographic_section)
        print(f'Infographic Section: {infographic_section_obj[0]}')

    # with CodeTimer('Response Generation', unit='s'):
    #     generated_response = generate_text(prompt_template)
    # print(f'Response: {generated_response[0]}')
