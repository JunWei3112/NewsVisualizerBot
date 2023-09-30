from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import transformers
import torch
import config

if __name__ == '__main__':
    model = 'meta-llama/Llama-2-7b-chat-hf'
    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model, token=config.HUGGING_FACE_ACCESS_TOKEN)

    stop_token_ids = [
        tokenizer.convert_tokens_to_ids(x) for x in [
            ['</s>'], ['User', ':'], ['Assistant', ':'],
            [tokenizer.convert_ids_to_tokens([9427])[0], ':']
        ]
    ]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = StoppingCriteriaList([StopOnTokens()])

    generate_text = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        stopping_criteria=stopping_criteria,
        max_new_tokens=256,
        token=config.HUGGING_FACE_ACCESS_TOKEN,
        temperature=0.1
    )

    instruction = "Append another graph to the infographic"

    prompt_template = f"""
<s>[INST] <<SYS>>
The user will provide an instruction to modify an infographic.
From the instruction provided by the user, extract the necessary information required to fill up the expected output.
The expected output should be formatted in the following schema:
(
    InstructionType: string // [ADD] if the instruction adds a new element to an infographic, [EDIT] if the instruction modifies an attribute of an existing element in the infographic, [DELETE] removes an existing element from an infographic.
    TargetElement: string // element that is to be added, edited or deleted
)
<</SYS>>
    
The instruction is as follows: {instruction} [/INST]
    """

    generated_sequences = generate_text(prompt_template)
    for sequence in generated_sequences:
        print(f"Result: {sequence['generated_text']}")

