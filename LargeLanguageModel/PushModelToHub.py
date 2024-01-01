from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import config

def push_model_to_hub(model_path):
    # This is the model that is stored in local disk / compute cluster
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.push_to_hub('flan-t5-large-instruction-type-tuned', use_auth_token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

def push_tokenizer_to_hub(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.push_to_hub('flan-t5-large-instruction-type-tuned', use_auth_token=config.HUGGING_FACE_ACCESS_TOKEN_WRITE)

if __name__ == '__main__':
    model_path = 'google/flan-t5-large-instruction-type-tuned'
    push_tokenizer_to_hub(model_path)
