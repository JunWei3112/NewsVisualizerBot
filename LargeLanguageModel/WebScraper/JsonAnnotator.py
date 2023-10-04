import json

def convert_json_to_posts():
    file_obj = open("posts.json", "r")
    json_content = file_obj.read()
    return json.loads(json_content)

def convert_json_to_annotated_ids():
    file_obj = open("annotated_ids.json", "r")
    json_content = file_obj.read()
    return json.loads(json_content)

def convert_json_to_annotations():
    file_obj = open("annotations.json", "r")
    json_content = file_obj.read()
    return json.loads(json_content)

def actions_on_exit(annotated_ids_json, annotations_json):
    json_string_ids = json.dumps(annotated_ids_json)
    json_file_ids = open("annotated_ids.json", "w")
    json_file_ids.write(json_string_ids)
    json_string_annotations = json.dumps(annotations_json)
    json_file_annotations = open('annotations.json', "w")
    json_file_annotations.write(json_string_annotations)
    json_file_ids.close()
    json_file_annotations.close()

def count_number_of_annotated_posts(annotated_ids_json):
    count = 0
    for annotated_id_json in annotated_ids_json:
        if annotated_id_json['action'] == 'Accepted':
            count += 1
    return count

if __name__ == '__main__':
    # The values for 'action' can be 'Accepted' and 'Rejected'
    annotated_ids = list()
    annotated_ids_json = convert_json_to_annotated_ids()
    for annotated_id_json in annotated_ids_json:
        annotated_ids.append(annotated_id_json['id'])

    annotations_json = convert_json_to_annotations()
    is_exit = False

    posts_json_obj = convert_json_to_posts()
    for post_json in posts_json_obj:
        if is_exit:
            break

        if post_json['id'] in annotated_ids:
            continue

        print('-------------------------------')
        print('Title: ' + post_json['Title'])

        annotated_id_json = {
            "id": post_json['id']
        }

        # To decide if the title of the post is usable
        is_rejected = input('Reject this? [Y]es / [N]o\n')
        is_rejected_bool = False
        if str.lower(is_rejected) == 'y' or str.lower(is_rejected) == 'yes':
            annotated_id_json["action"] = 'Rejected'
            is_rejected_bool = True
        elif str.lower(is_rejected) == 'n' or str.lower(is_rejected) == 'no':
            annotated_id_json["action"] = 'Accepted'
        else:
            actions_on_exit(annotated_ids_json, annotations_json)
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
            actions_on_exit(annotated_ids_json, annotations_json)
            is_exit = True
        else:
            annotation_json["instruction"]["prompt"] = instruction
        if is_exit:
            break

        while True:
            instruction_type = input('What is the instruction type? add / delete / edit / move \n' +
                                     'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(instruction_type) == 'exit' or str.lower(instruction_type) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json)
                is_exit = True
                break
            elif str.lower(instruction_type) == 'add' or str.lower(instruction_type) == 'delete' or str.lower(instruction_type) == 'edit' or str.lower(instruction_type) == 'move':
                annotation_json["annotation"]["instruction_type"] = instruction_type
                break
        if is_exit:
            break

        target_element = input('What is the target element? \n' +
                               'Type "exit" or "Exit" or "e" or "E" to exit.\n')
        if str.lower(target_element) == 'exit' or str.lower(target_element) == 'e':
            actions_on_exit(annotated_ids_json, annotations_json)
            is_exit = True
        else:
            annotation_json["annotation"]["target_element"] = target_element
        if is_exit:
            break

        element_value = input('What is the value of the element? \n' +
                              'If not specified, type "NONE"\n' +
                              'Type "exit" or "Exit" or "e" or "E" to exit.\n')
        if str.lower(element_value) == 'exit' or str.lower(element_value) == 'e':
            actions_on_exit(annotated_ids_json, annotations_json)
            is_exit = True
        elif str.lower(element_value) == 'none':
            annotation_json["annotation"]["element_value"] = 'NONE'
        else:
            annotation_json["annotation"]["element_value"] = element_value
        if is_exit:
            break

        if str.lower(instruction_type) == 'add' or str.lower(instruction_type) == 'move':
            target_location = input('What is the target location? If not specified, type "NONE"\n' +
                                    'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(target_location) == 'exit' or str.lower(target_location) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json)
                is_exit = True
            elif str.lower(element_value) == 'none':
                annotation_json["annotation"]["target_location"] = 'NONE'
            else:
                annotation_json["annotation"]["target_location"] = target_location
            if is_exit:
                break

            if str.lower(instruction_type) == 'move':
                element_location = input('What is the original location of the element? If not specified, type "NONE"\n' +
                                         'Type "exit" or "Exit" or "e" or "E" to exit.\n')
                if str.lower(target_location) == 'exit' or str.lower(target_location) == 'e':
                    actions_on_exit(annotated_ids_json, annotations_json)
                    is_exit = True
                elif str.lower(element_value) == 'none':
                    annotation_json["annotation"]["element_location"] = 'NONE'
                else:
                    annotation_json["annotation"]["element_location"] = element_location
        elif str.lower(instruction_type) == 'edit':
            edit_attribute = input('What is the attribute type being edited? SIZE / CONTENT / COLOR \n' +
                                   'If not specified, type "NONE"\n' +
                                   'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(edit_attribute) == 'exit' or str.lower(edit_attribute) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json)
                is_exit = True
            else:
                annotation_json["annotation"]["edit_attribute"] = edit_attribute
            if is_exit:
                break

            element_location = input('What is the original location of the element? If not specified, type "NONE"\n' +
                                     'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(target_location) == 'exit' or str.lower(target_location) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json)
                is_exit = True
            elif str.lower(element_value) == 'none':
                annotation_json["annotation"]["element_location"] = 'NONE'
            else:
                annotation_json["annotation"]["element_location"] = element_location
            if is_exit:
                break

            target_value = input('What is the target value? \n' +
                                 'If not specified, type "NONE"\n' +
                                 'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(target_value) == 'exit' or str.lower(target_value) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json)
                is_exit = True
            else:
                annotation_json["annotation"]["target_value"] = target_value
        else:
            element_location = input('What is the original location of the element? If not specified, type "NONE"\n' +
                                     'Type "exit" or "Exit" or "e" or "E" to exit.\n')
            if str.lower(target_location) == 'exit' or str.lower(target_location) == 'e':
                actions_on_exit(annotated_ids_json, annotations_json)
                is_exit = True
            elif str.lower(element_value) == 'none':
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

    actions_on_exit(annotated_ids_json, annotations_json)
