from datasets import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from linetimer import CodeTimer

def fine_tune_instruction_type():
    model_path = 'google/flan-t5-large-instruction-type-tuned'
    input_column = 'instruction'
    output_column = 'instruction_type'
    prompt_template = f'For the following instruction to modify an infographic ({{input}}), what is the instruction type (ADD/DELETE/EDIT/MOVE)?'

    with CodeTimer('Load Dataset', unit='s'):
        dataset = load_dataset("McSpicyWithMilo/infographic-instructions", data_files='instructions_types.json')

    with CodeTimer('Load Model', unit='s'):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    with CodeTimer('Load Tokenizer', unit='s'):
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    prefix = 'answer the question: '
    def pre_processing_function(sample):
        inputs = [prefix + prompt_template.format(input=item) for item in sample[input_column]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        labels = tokenizer(text_target=sample[output_column], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(pre_processing_function, batched=True)

    with CodeTimer('Load Trainer', unit='s'):
        output_dir = './results'
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=3e-4,
            num_train_epochs=5,
            weight_decay=0.01,
            save_strategy="no",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    with CodeTimer('Training Of The Model', unit='s'):
        trainer.train()

    with CodeTimer('Saving Model and Tokenizer', unit='s'):
        trainer.model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

if __name__ == '__main__':
    fine_tune_instruction_type()
