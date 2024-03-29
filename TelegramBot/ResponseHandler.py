import config
from linetimer import CodeTimer
import requests

def identify_instruction_type(instruction):
    with CodeTimer('Identify Instruction Type', unit='s'):
        headers = {"Authorization": f"Bearer {config.HUGGING_FACE_ACCESS_TOKEN}"}

        def query(payload):
            response = requests.post(config.INSTRUCTION_TYPE_API_URL, headers=headers, json=payload)
            return response.json()

        prompt_instruction_type = f'Given the following instruction to modify an infographic: "{instruction}" Identify the type of operation specified in the instruction: ADD/DELETE/EDIT/MOVE.'
        output_instruction_type = query({
            "inputs": prompt_instruction_type,
            "options": {
                "wait_for_model": True
            },
        })

    print(output_instruction_type)
    instruction_type = output_instruction_type[0]['generated_text']
    return instruction_type

def identify_target_element(instruction, instruction_type):
    with CodeTimer('Identify Target Element', unit='s'):
        headers = {"Authorization": f"Bearer {config.HUGGING_FACE_ACCESS_TOKEN}"}

        def query(payload):
            response = requests.post(config.TARGET_ELEMENT_API_URL, headers=headers, json=payload)
            return response.json()

        if instruction_type == 'ADD':
            prompt_target_element = f'Given the instruction, ({instruction}) Output the specific content or element that is added to the infographic. Ensure that the answer does not include the target location of the element or text.'
        elif instruction_type == 'DELETE':
            prompt_target_element = f'Given the instruction, ({instruction}) Output the specific content or element that is deleted from the infographic. Ensure that the answer does not include the target location of the element or text.'
        elif instruction_type == 'EDIT':
            prompt_target_element = f'Given the instruction, ({instruction}) Output the specific content or element that is modified in the infographic. Ensure that the answer does not include the target location of the element or text.'
        else:
            prompt_target_element = f'Given the instruction, ({instruction}) Output the specific content or element that is moved within the infographic. Ensure that the answer does not include the target location of the element or text.'
        output_target_element = query({
            "inputs": prompt_target_element,
            "options": {
                "wait_for_model": True
            },
        })

    print(output_target_element)
    target_element = output_target_element[0]['generated_text']
    return target_element

def identify_infographic_section(instruction, instruction_type):
    with CodeTimer('Identify Infographic Section', unit='s'):
        headers = {"Authorization": f"Bearer {config.HUGGING_FACE_ACCESS_TOKEN}"}

        def query(payload):
            response = requests.post(config.INFOGRAPHIC_SECTION_API_URL, headers=headers, json=payload)
            return response.json()

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
        output_infographic_section = query({
            "inputs": prompt_infographic_section,
            "options": {
                "wait_for_model": True
            },
        })

    print(output_infographic_section)
    infographic_section = output_infographic_section[0]['generated_text']
    return infographic_section

def identify_target_location(instruction):
    with CodeTimer('Identify Target Location', unit='s'):
        headers = {"Authorization": f"Bearer {config.HUGGING_FACE_ACCESS_TOKEN}"}

        def query(payload):
            response = requests.post(config.TARGET_LOCATION_API_URL, headers=headers, json=payload)
            return response.json()

        prompt_target_location = f'Identify the target location in the existing infographic where the new element should be placed based on the following user instruction: {{input}}'
        output_target_location = query({
            "inputs": prompt_target_location,
            "options": {
                "wait_for_model": True
            },
        })

    print(output_target_location)
    target_location = output_target_location[0]['generated_text']
    return target_location

def identify_target_element_new(instruction):
    with CodeTimer('Identify Target Element', unit='s'):
        headers = {"Authorization": f"Bearer {config.HUGGING_FACE_ACCESS_TOKEN}"}

        def query(payload):
            response = requests.post(config.TARGET_ELEMENT_NEW_API_URL, headers=headers, json=payload)
            return response.json()

        prompt_target_element = f'Given the instruction, ({instruction}) Output the specific content or element that is added to the infographic. Ensure that the answer does not include the target location of the element or text.'
        output_target_element = query({
            "inputs": prompt_target_element,
            "options": {
                "wait_for_model": True
            },
        })

    print(output_target_element)
    target_element = output_target_element[0]['generated_text']
    return target_element

def identify_target_location_new(instruction):
    with CodeTimer('Identify Target Location', unit='s'):
        headers = {"Authorization": f"Bearer {config.HUGGING_FACE_ACCESS_TOKEN}"}

        def query(payload):
            response = requests.post(config.TARGET_LOCATION_NEW_API_URL, headers=headers, json=payload)
            return response.json()

        prompt_target_location = f'Identify the target location in the existing infographic where the new element will be added next to based on the following user instruction: {instruction}'
        output_target_location = query({
            "inputs": prompt_target_location,
            "options": {
                "wait_for_model": True
            },
        })

    print(output_target_location)
    target_location = output_target_location[0]['generated_text']
    return target_location

def identify_infographic_section_new(location):
    with CodeTimer('Identify Infographic Section', unit='s'):
        headers = {"Authorization": f"Bearer {config.HUGGING_FACE_ACCESS_TOKEN}"}

        def query(payload):
            response = requests.post(config.INFOGRAPHIC_SECTION_NEW_API_URL, headers=headers, json=payload)
            return response.json()

        infographic_sections = "The infographic comprises a 'Header' section featuring the title of the article. The 'Number of Shares' section displays the numerical count of shares of this infographic. The 'Vote on Reliability' section presents a diagram reflecting user opinions and votes on the news article's reliability. The 'Related Facts' section lists statements and relevant facts related to the article, while the 'Latest Comments' section displays user-submitted comments on this infographic. The 'Knowledge Graph Summaries' section showcases sentiments towards various entities mentioned in the news through a knowledge graph. Lastly, 'Similar Articles' provides a list of similar articles with diverse viewpoints, each accompanied by a QR code, header, and a brief summary."
        task = f'Based on the context, which infographic section (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles) does "{location}" belong to?'
        output_infographic_section = query({
            "inputs": {
                "question": task,
                "context": infographic_sections
            },
            "options": {
                "wait_for_model": True
            },
        })

    print(output_infographic_section)
    return remove_quotes(output_infographic_section['answer'])

def remove_quotes(string):
    if string.startswith(' \'') and string.endswith('\''):
        return string[2:-1]
    else:
        return string

def generate_intermediate_representation(instruction):
    instruction_type = identify_instruction_type(instruction)
    intermediate_representation = f'Instruction Type: {instruction_type}\n'
    target_element = identify_target_element(instruction, instruction_type)
    intermediate_representation += f'Target Element: {target_element}\n'
    infographic_section = identify_infographic_section(instruction, instruction_type)
    intermediate_representation += f'Infographic Section: {infographic_section}\n'
    if instruction_type == "ADD":
        target_location = identify_target_location(instruction)
        intermediate_representation += f'Target Location: {target_location}'
    return intermediate_representation

def generate_intermediate_representation_new(instruction):
    instruction_type = identify_instruction_type(instruction)
    intermediate_representation = f'Instruction Type: {instruction_type}\n'
    target_element = identify_target_element_new(instruction)
    intermediate_representation += f'Target Element: {target_element}\n'
    target_location = identify_target_location_new(instruction)
    intermediate_representation += f'Target Location: {target_location}\n'
    infographic_section = identify_infographic_section_new(target_location)
    intermediate_representation += f'Infographic Section: {infographic_section}\n'
    return intermediate_representation

if __name__ == '__main__':
    instruction = "weave in an additional relevant fact close to the similar articles section"
    intermediate_rep = generate_intermediate_representation_new(instruction)
    print(intermediate_rep)
