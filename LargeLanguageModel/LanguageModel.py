from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch
import config

if __name__ == '__main__':
    model_path = 'meta-llama/Llama-2-7b-chat-hf'
    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = LlamaForCausalLM.from_pretrained(model_path, token=config.HUGGING_FACE_ACCESS_TOKEN)
    tokenizer = AutoTokenizer.from_pretrained(model_path, token=config.HUGGING_FACE_ACCESS_TOKEN)

    generate_text = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        max_new_tokens=256,
        token=config.HUGGING_FACE_ACCESS_TOKEN,
        temperature=0.1
    )

    instruction = "Delete the header"

    prompt_template = f"""
<s>[INST] <<SYS>>
The user will provide an instruction to modify an infographic.
From the instruction provided by the user, extract the necessary information required to fill up the expected output.
If the instruction adds a new element to an infographic, the expected output should be formatted in the following schema:
(
    InstructionType: ADD
    TargetElement: string // element that is to be added
    ElementValue: string // value of the element that is to be added, if specified in the instruction. If not specified, set the value as NONE
    TargetLocation: string // location of the new element, if specified in the instruction. If not specified, set the value to NONE
)
If the instruction modifies an attribute of an existing element in the infographic, the expected output should be formatted in the following schema:
(
    InstructionType: EDIT
    TargetElement: string // element that is to be modified
    ElementValue: string // value of the element that is to be modified, if specified in the instruction. If not specified, set the value as NONE
    EditAttribute: string // [SIZE] if the size of the element is modified, [CONTENT] if the content of the element is modified
)
If the instruction modifies the location of an existing element in an infographic, the expected output should be formatted in the following schema:
(
    InstructionType: MOVE
    TargetElement: string // element that is to be moved
    ElementValue: string // value of the element that is to be moved, if specified in the instruction. If not specified, set the value as NONE
    TargetLocation: string // new location of the element, if specified in the instruction
)
If the instruction removes an existing element from an infographic, the expected output should be formatted in the following schema:
(
    InstructionType: DELETE
    TargetElement: string // element that is to be deleted
    ElementValue: string // value of the element that is to be deleted, if specified in the instruction. If not specified, set the value as NONE
)
<</SYS>>

The instruction is as follows: Add new comment: "This is a new comment" [/INST]
(
    InstructionType: ADD
    TargetElement: comment
    ElementValue: This is a new comment
    TargetLocation: NONE
)</s><s> [INST]

The instruction is as follows: Remove the second article in 'Similar Articles' [/INST]
(
    InstructionType: DELETE
    TargetElement: article
    ElementValue: NONE
)</s><s> [INST]

The instruction is as follows: I want the knowledge graph to be on the right of the news links' [/INST]
(
    InstructionType: MOVE
    TargetElement: knowledge graph
    ElementValue: NONE
    TargetLocation: right of the news links
)</s><s> [INST]

The instruction is as follows: Make the knowledge graph larger [/INST]
(
    InstructionType: EDIT
    TargetElement: knowledge graph
    ElementValue: NONE
    EditAttribute: SIZE
)</s><s> [INST]

The instruction is as follows: {instruction} [/INST]"""

    generated_sequences = generate_text(prompt_template)
    for sequence in generated_sequences:
        print(f"Result: {sequence['generated_text']}")

