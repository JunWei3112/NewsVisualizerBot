from transformers import AutoTokenizer

def count_tokens(prompt):
    model_path = 'google/flan-t5-large'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenized_prompt = tokenizer.tokenize(prompt)
    token_counts = len(tokenized_prompt)
    return token_counts

if __name__ == '__main__':
    prompt = "Section 2) Number of Shares: This section shows the number of times the news link has been shared to the chatbot. It contains a title labeled 'Number of Shares' and a numerical text of the number of shares."
    token_counts = count_tokens(prompt)
    print(f"Token Count: {token_counts}")
