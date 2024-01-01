import json


def get_json_obj_from_file(file_name):
    file_obj = open(file_name, "r")
    json_content = file_obj.read()
    return json.loads(json_content)


def actions_on_exit(annotated_ids_json, annotations_json, ids_json_file_name, annotations_json_file_name):
    json_string_ids = json.dumps(annotated_ids_json)
    json_file_ids = open(ids_json_file_name, "w")
    json_file_ids.write(json_string_ids)
    json_string_annotations = json.dumps(annotations_json)
    json_file_annotations = open(annotations_json_file_name, "w")
    json_file_annotations.write(json_string_annotations)
    json_file_ids.close()
    json_file_annotations.close()


def count_number_of_annotated_posts(annotated_ids_json):
    count = 0
    for annotated_id_json in annotated_ids_json:
        if annotated_id_json['action'] == 'Accepted':
            count += 1
    return count


def annotate_instructions(ids_json_file, annotations_json_file, posts_file):
    # The values for 'action' can be 'Accepted' and 'Rejected'
    annotated_ids = list()
    annotated_ids_json = get_json_obj_from_file(ids_json_file)
    for annotated_id_json in annotated_ids_json:
        annotated_ids.append(annotated_id_json['id'])

    annotations_json = get_json_obj_from_file(annotations_json_file)
    is_exit = False

    instructions_json_obj = get_json_obj_from_file(posts_file)
    for instruction_json in instructions_json_obj:
        if is_exit:
            break

        if instruction_json['id'] in annotated_ids:
            continue

        print('-------------------------------')
        if 'instruction' not in instruction_json:
            print('Title: ' + instruction_json['Title'])
        else:
            print('Instruction: ' + instruction_json['instruction'])

        annotated_id_json = {
            "id": instruction_json['id']
        }

        # To decide if the title of the post is usable
        is_accepted = input('Accept this? [Y]es / [N]o\n')
        is_rejected_bool = False
        if str.lower(is_accepted) == 'n' or str.lower(is_accepted) == 'no':
            annotated_id_json["action"] = 'Rejected'
            is_rejected_bool = True
        elif str.lower(is_accepted) == 'y' or str.lower(is_accepted) == 'yes':
            annotated_id_json["action"] = 'Accepted'
        else:
            actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
            is_exit = True
            break
        if is_rejected_bool:
            annotated_ids_json.append(annotated_id_json)
            continue

        annotation_json = {
            "instruction": {},
            "annotation": {}
        }

        instruction = input('What is the revised instruction? Type "exit" or "Exit" or "e" or "E" to exit.\n')
        if str.lower(instruction) == 'exit' or str.lower(instruction) == 'e':
            actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
            is_exit = True
        else:
            annotation_json["instruction"]["prompt"] = instruction
        if is_exit:
            break

        while True:
            instruction_type = input('What is the instruction type? add / delete / edit / move \n' +
                                     'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(instruction_type) == 'exit' or str.lower(instruction_type) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
                is_exit = True
                break
            elif str.lower(instruction_type) == 'add' or str.lower(instruction_type) == 'delete' or str.lower(
                    instruction_type) == 'edit' or str.lower(instruction_type) == 'move':
                annotation_json["annotation"]["instruction_type"] = str.upper(instruction_type)
                break
        if is_exit:
            break

        target_element = input('What is the target element? \n' +
                               'Type "exit" or "Exit" or "e" or "E" to exit.\n')
        if str.lower(target_element) == 'exit' or str.lower(target_element) == 'e':
            actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
            is_exit = True
        else:
            annotation_json["annotation"]["target_element"] = target_element
        if is_exit:
            break

        infographic_section = input('What is the infographic section? 1) Number of Shares, 2) Vote on Reliability, 3) Related Facts, 4) Latest Comments, 5) Knowledge Graph Summaries, 6) Similar Articles, 7) Header\n' +
                                    'Type "exit" or "Exit" or "e" or "E" to exit.\n')
        if str.lower(infographic_section) == 'exit' or str.lower(infographic_section) == 'e':
            actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
            is_exit = True
        elif str.lower(infographic_section) == 'none':
            annotation_json["annotation"]["infographic_section"] = 'NONE'
        else:
            annotation_json["annotation"]["infographic_section"] = infographic_section
        if is_exit:
            break

        if str.lower(instruction_type) == 'add' or str.lower(instruction_type) == 'move':
            if str.lower(instruction_type) == 'move':
                element_location = input(
                    'What is the original location of the element? If not specified, type "NONE"\n' +
                    'Type "exit" or "Exit" or "e" or "E" to exit.\n')
                if str.lower(element_location) == 'exit' or str.lower(element_location) == 'e':
                    actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
                    is_exit = True
                elif str.lower(element_location) == 'none':
                    annotation_json["annotation"]["element_location"] = 'NONE'
                else:
                    annotation_json["annotation"]["element_location"] = element_location
            if is_exit:
                break

            target_location = input('What is the target location? If not specified, type "NONE"\n' +
                                    'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(target_location) == 'exit' or str.lower(target_location) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
                is_exit = True
            elif str.lower(target_location) == 'none':
                annotation_json["annotation"]["target_location"] = 'NONE'
            else:
                annotation_json["annotation"]["target_location"] = target_location
            if is_exit:
                break
        elif str.lower(instruction_type) == 'edit':
            element_location = input('What is the original location of the element? If not specified, type "NONE"\n' +
                                     'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(element_location) == 'exit' or str.lower(element_location) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
                is_exit = True
            elif str.lower(element_location) == 'none':
                annotation_json["annotation"]["element_location"] = 'NONE'
            else:
                annotation_json["annotation"]["element_location"] = element_location
            if is_exit:
                break

            edit_attribute = input('What is the attribute type being edited? SIZE / CONTENT \n' +
                                   'If not specified, type "NONE"\n' +
                                   'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(edit_attribute) == 'exit' or str.lower(edit_attribute) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
                is_exit = True
            else:
                annotation_json["annotation"]["edit_attribute"] = str.upper(edit_attribute)
            if is_exit:
                break

            target_value = input('What is the target value? \n' +
                                 'If not specified, type "NONE"\n' +
                                 'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(target_value) == 'exit' or str.lower(target_value) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
                is_exit = True
            else:
                annotation_json["annotation"]["target_value"] = target_value
        else:
            element_location = input('What is the original location of the element? If not specified, type "NONE"\n' +
                                     'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(element_location) == 'exit' or str.lower(element_location) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)
                is_exit = True
            elif str.lower(element_location) == 'none':
                annotation_json["annotation"]["element_location"] = 'NONE'
            else:
                annotation_json["annotation"]["element_location"] = element_location
        if is_exit:
            break

        annotated_ids_json.append(annotated_id_json)
        annotations_json.append(annotation_json)

    print('-------------------------------')
    print('NO MORE POSTS LEFT TO ANNOTATE')
    print('-------------------------------')

    print('Number of annotated posts: ' + str(count_number_of_annotated_posts(annotated_ids_json)))
    print('Total number of posts: ' + str(len(annotated_ids_json)))

    actions_on_exit(annotated_ids_json, annotations_json, ids_json_file, annotations_json_file)


def generate_unique_ids(instructions_file_name):
    instructions = get_json_obj_from_file(instructions_file_name)

    max_instruction_count = 0
    for instruction in instructions:
        if 'id' in instruction:
            max_instruction_count = max(max_instruction_count, instruction["id"])

    for instruction in instructions:
        if 'id' not in instruction:
            instruction["id"] = max_instruction_count + 1
            max_instruction_count += 1

    json_string = json.dumps(instructions)
    json_file_ids = open(instructions_file_name, "w")
    json_file_ids.write(json_string)


if __name__ == '__main__':
    llm_instructions_file = 'bard_data_files/llm_generated_instructions.json'
    llm_ids_file = 'bard_data_files/annotated_ids.json'
    llm_annotations_file = 'bard_data_files/annotations.json'
    annotate_instructions(llm_ids_file, llm_annotations_file, llm_instructions_file)
    # generate_unique_ids(llm_instructions_file)
