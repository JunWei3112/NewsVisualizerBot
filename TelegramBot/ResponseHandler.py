import config
from linetimer import CodeTimer
import requests

def generate_intermediate_representation(instruction):
    with CodeTimer('Generate Intermediate Representation', unit='s'):
        headers = {"Authorization": f"Bearer {config.HUGGING_FACE_ACCESS_TOKEN}"}

        def query(payload):
            response = requests.post(config.INFERENCE_API_URL, headers=headers, json=payload)
            return response.json()

        prompt_instruction_type = f'For the following instruction to modify an infographic ({instruction}), what is the instruction type (ADD/DELETE/EDIT/MOVE)?'
        output_instruction_type = query({
            "inputs": prompt_instruction_type,
            "wait_for_model": True
        })

    print(output_instruction_type)
    instruction_type = output_instruction_type[0]['generated_text']
    return instruction_type
