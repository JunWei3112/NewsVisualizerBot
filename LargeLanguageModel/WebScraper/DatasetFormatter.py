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
#     annotations_json_file_name = 'bard_data_files/annotations.json'
#     annotations = load_annotations(annotations_json_file_name)
#
#     dataset = list()
#
#     instruction = f'''
# The user will provide an instruction to modify an infographic.
#
# InstructionType: ADD, EDIT, MOVE or DELETE
# TargetElement: string // element that is to be added/edited/moved/deleted
# InfographicSection: string // The infographic has the following sections: 1) Number of Shares, 2) Vote on Reliability, 3) Related Facts, 4) Latest Comments, 5) Knowledge Graph Summaries, 6) Similar Articles, 7) Header. Try to infer the infographic section that the element will be added/edited/moved/deleted from, and set it as the value of InfographicSection. If you are unable to infer the section, set the value as NONE
# ElementLocation: string // location of the target element in the infographic section, if specified in the instruction. If not specified, set the value as NONE
# TargetLocation: string // new location of the target element, if specified in the instruction. If not specified, set the value as NONE
#
# From the instruction provided by the user, extract the necessary information required to fill up the expected output. Do not generate any fields that are not included in the expected output
#
# If the instruction adds a new element to an infographic, the expected output should be formatted in the following schema:
# (
#     InstructionType: ADD
#     TargetElement: string
#     InfographicSection: string
#     TargetLocation: string
# )
# If the instruction modifies an attribute of an existing element in the infographic, the expected output should be formatted in the following schema:
# (
#     InstructionType: EDIT
#     TargetElement: string
#     InfographicSection: string
#     ElementLocation: string
#     EditAttribute: string // [SIZE] if the size of the element is modified, [CONTENT] if the textual content of the element is modified
#     TargetValue: string // new size or textual content of the element, if specified in the instruction. If not specified, set the value as NONE
# )
# If the instruction modifies the location of an existing element in an infographic, the expected output should be formatted in the following schema:
# (
#     InstructionType: MOVE
#     TargetElement: string
#     InfographicSection: string
#     ElementLocation: string
#     TargetLocation: string
# )
# If the instruction removes an existing element from an infographic, the expected output should be formatted in the following schema:
# (
#     InstructionType: DELETE
#     TargetElement: string
#     InfographicSection: string
#     ElementLocation: string
# )
# '''
#
#     for annotation in annotations:
#         annotation_instruction = annotation["instruction"]
#         instruction_json = {
#             "instruction": instruction,
#             "context": annotation_instruction["prompt"]
#         }
#
#         self_annotation = annotation["annotation"]
#         instruction_type = self_annotation["instruction_type"]
#
#         if instruction_type == 'ADD':
#             response = f'''
# (
#     InstructionType: ADD
#     TargetElement: {self_annotation["target_element"]}
#     InfographicSection: {self_annotation["infographic_section"]}
#     TargetLocation: {self_annotation["target_location"]}
# )
# '''
#             instruction_json["response"] = response
#         elif instruction_type == 'DELETE':
#             response = f'''
# (
#     InstructionType: DELETE
#     TargetElement: {self_annotation["target_element"]}
#     InfographicSection: {self_annotation["infographic_section"]}
#     ElementLocation: {self_annotation["element_location"]}
# )
# '''
#             instruction_json["response"] = response
#         elif instruction_type == 'EDIT':
#             response = f'''
# (
#     InstructionType: EDIT
#     TargetElement: {self_annotation["target_element"]}
#     InfographicSection: {self_annotation["infographic_section"]}
#     ElementLocation: {self_annotation["element_location"]}
#     EditAttribute: {self_annotation["edit_attribute"]}
#     TargetValue: {self_annotation["target_value"]}
# )
# '''
#             instruction_json["response"] = response
#         else:
#             response = f'''
# (
#     InstructionType: MOVE
#     TargetElement: {self_annotation["target_element"]}
#     InfographicSection: {self_annotation["infographic_section"]}
#     ElementLocation: {self_annotation["element_location"]}
#     TargetLocation: {self_annotation["target_location"]}
# )
# '''
#             instruction_json["response"] = response
#
#         dataset.append(instruction_json)
#
#     dataset_json_file_name = 'bard_data_files/dataset.json'
#     save_dataset(dataset_json_file_name, dataset)

