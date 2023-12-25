import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoTokenizer, LlamaForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
import config
import time
import os

def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["instruction"])):
        instruction = examples["instruction"][i]
        context = 'The instruction is as follows: ' + examples["context"][i]
        response = examples["response"][i]

        text = f'### Instruction: {instruction}\n ### Context: {context}\n ### Response: {response}'
        output_texts.append(text)
    return output_texts

if __name__ == '__main__':
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    dataset = load_dataset("McSpicyWithMilo/infographic-instructions", split="train")

    print('---------------------------------')
    print('Dataset loaded')
    print('---------------------------------')

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    model_path = 'meta-llama/Llama-2-7b-chat-hf-fine-tuned-2'
    print('INPUT MODEL PATH IS ' + model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map=device_map,
        token=config.HUGGING_FACE_ACCESS_TOKEN)
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    output_dir = "./results"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=500
    )

    torch.cuda.empty_cache()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        peft_config=peft_config,
        train_dataset=dataset,
        formatting_func=formatting_prompts_func,
        max_seq_length=512,
        args=training_args,
    )

    print('---------------------------------')
    print('Trainer set up')
    print('---------------------------------')

    torch.cuda.empty_cache()

    start_time = time.time()

    trainer.train()

    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Time taken to train: {time_taken}")

    torch.cuda.empty_cache()

    output_model_path = 'meta-llama/Llama-2-7b-chat-hf-fine-tuned-2'
    print('OUTPUT MODEL PATH IS ' + output_model_path)
    trainer.save_model(output_model_path)

    print('---------------------------------')
    print('Model and tokenizer saved')
    print('---------------------------------')
