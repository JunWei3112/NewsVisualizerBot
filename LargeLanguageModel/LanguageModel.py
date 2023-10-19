from transformers import AutoTokenizer, LlamaForCausalLM
import transformers
import torch
import config
from linetimer import CodeTimer

if __name__ == '__main__':
    model_path = 'meta-llama/Llama-2-7b-chat-hf'
    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    with CodeTimer('Model Loading'):
        model = LlamaForCausalLM.from_pretrained(model_path, token=config.HUGGING_FACE_ACCESS_TOKEN)

    with CodeTimer('Tokenizer Loading'):
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=config.HUGGING_FACE_ACCESS_TOKEN)

    with CodeTimer('Pipeline Loading'):
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

    instruction = "Delete the statement 'The news article is based on a scientific study' from the list of statements on related facts."

    prompt_template = f"""
<s>[INST] <<SYS>>
The user will provide an instruction to modify an infographic.

InstructionType: ADD, EDIT, MOVE or DELETE
TargetElement: string // element that is to be added/edited/moved/deleted
InfographicSection: string // The infographic has the following sections: 1) Number of Shares, 2) Vote on Reliability, 3) Related Facts, 4) Latest Comments, 5) Knowledge Graph Summaries, 6) Similar Articles, 7) Header. Try to infer the infographic section that the element will be added/edited/moved/deleted from, and set it as the value of InfographicSection. If you are unable to infer the section, set the value as NONE
ElementLocation: string // location of the target element in the infographic section, if specified in the instruction. If not specified, set the value as NONE
TargetLocation: string // new location of the target element, if specified in the instruction. If not specified, set the value as NONE

From the instruction provided by the user, extract the necessary information required to fill up the expected output. Do not generate any fields that are not included in the expected output

If the instruction adds a new element to an infographic, the expected output should be formatted in the following schema:
(
    InstructionType: ADD
    TargetElement: string
    InfographicSection: string
    TargetLocation: string
)
If the instruction modifies an attribute of an existing element in the infographic, the expected output should be formatted in the following schema:
(
    InstructionType: EDIT
    TargetElement: string
    InfographicSection: string
    ElementLocation: string
    EditAttribute: string // [SIZE] if the size of the element is modified, [CONTENT] if the textual content of the element is modified
    TargetValue: string // new size or textual content of the element, if specified in the instruction. If not specified, set the value as NONE
)
If the instruction modifies the location of an existing element in an infographic, the expected output should be formatted in the following schema:
(
    InstructionType: MOVE
    TargetElement: string
    InfographicSection: string
    ElementLocation: string
    TargetLocation: string
)
If the instruction removes an existing element from an infographic, the expected output should be formatted in the following schema:
(
    InstructionType: DELETE
    TargetElement: string
    InfographicSection: string
    ElementLocation: string
)
<</SYS>>

The instruction is as follows: Add new comment: "This is a new comment" [/INST]
(
    InstructionType: ADD
    TargetElement: "This is a new comment"
    InfographicSection: 4) Latest Comments
    TargetLocation: NONE
)</s><s> [INST]

The instruction is as follows: Remove the second article in 'Similar Articles' [/INST]
(
    InstructionType: DELETE
    TargetElement: second article
    InfographicSection: 6) Similar Articles
    ElementLocation: NONE
)</s><s> [INST]

The instruction is as follows: I want the knowledge graph to be on the right of the news links' [/INST]
(
    InstructionType: MOVE
    TargetElement: knowledge graph
    InfographicSection: 5) Knowledge Graph Summaries
    ElementLocation: NONE
    TargetLocation: right of the news links
)</s><s> [INST]

The instruction is as follows: Make the knowledge graph larger [/INST]
(
    InstructionType: EDIT
    TargetElement: knowledge graph
    InfographicSection: 5) Knowledge Graph Summaries
    ElementLocation: NONE
    EditAttribute: SIZE
    TargetValue: larger
)</s><s> [INST]

The instruction is as follows: {instruction} [/INST]"""

    with CodeTimer('Response Generation'):
        generated_sequences = generate_text(prompt_template)

    for sequence in generated_sequences:
        print(f"Result: {sequence['generated_text']}")
