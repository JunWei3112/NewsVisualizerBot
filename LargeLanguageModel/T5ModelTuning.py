from datasets import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from linetimer import CodeTimer

def format_instruction_types(dataset):
    features = dataset.features.copy()
    features['instruction_type'] = Value(dtype="string", id=None)
    dataset = dataset.cast(features)

    def modify_instruction_type_field(row):
        if row["instruction_type"] == '0':
            row["instruction_type"] = "ADD"
        elif row["instruction_type"] == '1':
            row["instruction_type"] = "DELETE"
        elif row["instruction_type"] == '2':
            row["instruction_type"] = "EDIT"
        elif row["instruction_type"] == '3':
            row["instruction_type"] = "MOVE"
        return row

    formatted_dataset = dataset.map(modify_instruction_type_field)
    return formatted_dataset

def fine_tune_instruction_type():
    input_model_path = 'google/flan-t5-large'
    output_model_path = 'google/flan-t5-large-instruction-type-200'
    input_column = 'instruction'
    output_column = 'instruction_type'
    prompt_template = f'For the following instruction to modify an infographic ({{input}}), what is the instruction type (ADD/DELETE/EDIT/MOVE)?'

    with CodeTimer('Load Dataset', unit='s'):
        dataset = load_dataset("McSpicyWithMilo/infographic-instructions",
                               data_files='instructions_200.json',
                               split="train")
        stratify_column_name = 'instruction_type'
        dataset = dataset.class_encode_column(stratify_column_name).train_test_split(test_size=0.2,
                                                                                     stratify_by_column=stratify_column_name)
        train_dataset = dataset["train"]
        train_dataset = format_instruction_types(train_dataset)
        test_dataset = dataset["test"]
        test_dataset = format_instruction_types(test_dataset)

    with CodeTimer('Load Model', unit='s'):
        model = AutoModelForSeq2SeqLM.from_pretrained(input_model_path)

    with CodeTimer('Load Tokenizer', unit='s'):
        tokenizer = AutoTokenizer.from_pretrained(input_model_path)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    prefix = 'answer the question: '
    def pre_processing_function(sample):
        inputs = [prefix + prompt_template.format(input=item) for item in sample[input_column]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        labels = tokenizer(text_target=sample[output_column], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = train_dataset.map(pre_processing_function, batched=True)
    tokenized_test_dataset = test_dataset.map(pre_processing_function, batched=True)

    with CodeTimer('Load Trainer', unit='s'):
        output_dir = './results'
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=3e-4,
            num_train_epochs=5,
            weight_decay=0.01,
            save_strategy="no",
            evaluation_strategy="epoch",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

    with CodeTimer('Training Of The Model', unit='s'):
        trainer.train()

    with CodeTimer('Evaluation Of The Model', unit='s'):
        results = trainer.evaluate()
        print(results)

    with CodeTimer('Saving Model and Tokenizer', unit='s'):
        trainer.model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_model_path)

if __name__ == '__main__':
    fine_tune_instruction_type()
