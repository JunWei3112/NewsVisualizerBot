from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import transformers
import torch
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import config

def check_for_gpu():
    print('------------------------------------')
    if torch.cuda.is_available():
        print("GPU is available!")
    else:
        print("GPU is not available!")
    print('------------------------------------')

if __name__ == '__main__':
    check_for_gpu()

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
        token=config.HUGGING_FACE_ACCESS_TOKEN
    )

    response_schemas = [
        ResponseSchema(name="instruction_type", description="[ADD] if it is an Add instruction, "
                       + "[EDIT] if it is an Edit instruction or [DELETE] if it is a Delete instruction"),
        ResponseSchema(name="target_element", description="element that is to be added, edited or deleted"),
        ResponseSchema(name="explanation", description="explanation behind the categorized instruction type to the user's instruction")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

    format_instructions = output_parser.get_format_instructions()
    prompt_template = """Definitions of Add, Edit and Delete instructions are provided in the context. 
    Identify whether the following instruction is an Add, Edit or Delete instruction: {instruction}
    
    Context: {context}
    Format Instructions: {format_instructions}"""
    context = """An Add instruction is one that adds a new element to an infographic.
    An Edit instruction is one that modifies an attribute of an element that already exists in an infographic.
    A Delete instruction is one that removes an existing element from an infographic."""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["instruction", "context"],
        partial_variables={"format_instructions": format_instructions}
    )

    llm = HuggingFacePipeline(pipeline=generate_text)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    output = llm_chain.predict(
        instruction='Take out the pie chart',
        context=context
    ).lstrip()
    print(output)

