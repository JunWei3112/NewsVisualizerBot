from datasets import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from linetimer import CodeTimer
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import torch
import transformers
import numpy as np

def generate_local_pipeline(model_path):
    device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    with CodeTimer('Model Loading', unit='s'):
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print('----------------------------')
        print(f'Model being evaluated: {model_path}')
        print('----------------------------')

    with CodeTimer('Tokenizer Loading', unit='s'):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.model_max_length = 2048

    with CodeTimer('Pipeline Loading', unit='s'):
        generate_text = transformers.pipeline(
            task="text2text-generation",
            model=model,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=256,
        )

    return generate_text

def fine_tune_instruction_type_lora(input_model_path, output_model_path):
    input_column = 'instruction'
    output_column = 'instruction_type'
    prompt_template = f'Given the following instruction to modify an infographic: "{{input}}" Identify the type of operation specified in the instruction: ADD/DELETE/EDIT/MOVE.'

    with CodeTimer('Load Dataset', unit='s'):
        disable_caching()
        dataset = load_dataset("McSpicyWithMilo/instruction-types-0.2split")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    with CodeTimer('Load Model', unit='s'):
        model = AutoModelForSeq2SeqLM.from_pretrained(input_model_path)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)

    with CodeTimer('Load Tokenizer', unit='s'):
        tokenizer = AutoTokenizer.from_pretrained(input_model_path)

    prefix = 'answer the question: '

    def pre_processing_function(sample):
        inputs = [prefix + prompt_template.format(input=item) for item in sample[input_column]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        labels = tokenizer(text_target=sample[output_column], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = train_dataset.map(pre_processing_function, batched=True)
    tokenized_test_dataset = test_dataset.map(pre_processing_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

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
            report_to="tensorboard",
        )

        print('-------------TRAINING HYPER-PARAMETERS----------------')
        print('learning_rate = 3e-4')
        print('num_train_epochs = 5')
        print('weight_decay = 0.01')
        print('-------------TRAINING HYPER-PARAMETERS----------------')

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

    with CodeTimer('Evaluation Of The Model (Using Trainer)', unit='s'):
        results = trainer.evaluate()
        print(results)

    with CodeTimer('Saving Model and Tokenizer', unit='s'):
        trainer.model.save_pretrained(output_model_path)
        trainer.model.base_model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_model_path)

    with CodeTimer('Evaluation Of The Model (Accuracy Metric)', unit='s'):
        evaluate_instruction_type_lora(output_model_path)

def evaluate_instruction_type_lora(model_path):
    disable_caching()
    eval_dataset = load_dataset("McSpicyWithMilo/instruction-types-0.2split", split="test")
    predictions, references = [], []
    generate_text = generate_local_pipeline(model_path)

    for row in eval_dataset:
        instruction = row["instruction"]
        instruction_type = row["instruction_type"]
        prompt = f'Given the following instruction to modify an infographic: "{instruction}" Identify the type of operation specified in the instruction: ADD/DELETE/EDIT/MOVE.'
        prediction_obj = generate_text(prompt)
        prediction = prediction_obj[0]['generated_text']
        predictions.append(prediction)
        references.append(instruction_type)

    prediction_to_int_mapping = {label: integer for integer, label in enumerate(set(predictions))}
    predictions = list(map(lambda x: prediction_to_int_mapping[x], predictions))
    references_to_int_mapping = {label: integer for integer, label in enumerate(set(references))}
    references = list(map(lambda x: references_to_int_mapping[x], references))
    accuracy_metric = evaluate.load("accuracy")
    results = accuracy_metric.compute(references=references, predictions=predictions)
    print('--------------EVALUATION RESULTS (INSTRUCTION TYPE)------------------')
    print(results)
    print('--------------EVALUATION RESULTS (INSTRUCTION TYPE)------------------')

def fine_tune_infographic_section_lora(input_model_path, output_model_path):
    input_column = 'instruction'
    output_column = 'infographic_section'
    instruction_type_column = 'instruction_type'
    infographic_sections = "The infographic comprises a 'Header' section featuring the title and a QR code linking to the content. The 'Number of Shares' section displays the numerical count of shares. The 'Vote on Reliability' section presents a diagram reflecting user opinions on the news article's reliability. The 'Related Facts' section lists statements related to the article, while the 'Latest Comments' section displays user-submitted comments. The 'Knowledge Graph Summaries' section showcases sentiments towards various entities mentioned in the news through a knowledge graph. Lastly, 'Similar Articles' provides a list of articles with diverse viewpoints, each accompanied by a QR code, header, and a brief summary. "

    with CodeTimer('Load Dataset', unit='s'):
        disable_caching()
        dataset = load_dataset("McSpicyWithMilo/infographic-sections-0.2split")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    with CodeTimer('Load Model', unit='s'):
        model = AutoModelForSeq2SeqLM.from_pretrained(input_model_path)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)

    with CodeTimer('Load Tokenizer', unit='s'):
        tokenizer = AutoTokenizer.from_pretrained(input_model_path)

    prefix = 'answer the question: '

    def pre_processing_function(sample):
        instruction_type = sample[instruction_type_column]
        if instruction_type == 'ADD':
            task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element will be added to within the existing infographic in response to '{{input}}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
        elif instruction_type == 'DELETE':
            task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element will be deleted from within the existing infographic in response to '{{input}}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
        elif instruction_type == 'EDIT':
            task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element will be edited or modified within the existing infographic in response to '{{input}}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
        else:
            task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element is originally in within the existing infographic in response to '{{input}}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
        prompt_template = infographic_sections + task

        inputs = [prefix + prompt_template.format(input=item) for item in sample[input_column]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        labels = tokenizer(text_target=sample[output_column], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = train_dataset.map(pre_processing_function, batched=True)
    tokenized_test_dataset = test_dataset.map(pre_processing_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    with CodeTimer('Load Trainer', unit='s'):
        output_dir = './results'
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=3e-4,
            num_train_epochs=10,
            weight_decay=1e-4,
            save_strategy="no",
            evaluation_strategy="epoch",
            report_to="tensorboard",
        )

        print('-------------TRAINING HYPER-PARAMETERS----------------')
        print('learning_rate = 3e-4')
        print('num_train_epochs = 10')
        print('weight_decay = 1e-4')
        print('-------------TRAINING HYPER-PARAMETERS----------------')

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_test_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator
        )

    with CodeTimer('Training Of The Model', unit='s'):
        trainer.train()

    with CodeTimer('Evaluation Of The Model (Using Trainer)', unit='s'):
        results = trainer.evaluate()
        print(results)

    with CodeTimer('Saving Model and Tokenizer', unit='s'):
        trainer.model.save_pretrained(output_model_path)
        trainer.model.base_model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_model_path)

    with CodeTimer('Evaluation Of The Model (Accuracy Metric)', unit='s'):
        evaluate_infographic_section_lora(output_model_path)

def evaluate_infographic_section_lora(model_path):
    disable_caching()
    eval_dataset = load_dataset("McSpicyWithMilo/infographic-sections-0.2split", split="test")
    predictions, references = [], []
    generate_text = generate_local_pipeline(model_path)

    for row in eval_dataset:
        instruction = row["instruction"]
        instruction_type = row["instruction_type"]
        infographic_section = row["infographic_section"]

        infographic_sections = "The infographic comprises a 'Header' section featuring the title and a QR code linking to the content. The 'Number of Shares' section displays the numerical count of shares. The 'Vote on Reliability' section presents a diagram reflecting user opinions on the news article's reliability. The 'Related Facts' section lists statements related to the article, while the 'Latest Comments' section displays user-submitted comments. The 'Knowledge Graph Summaries' section showcases sentiments towards various entities mentioned in the news through a knowledge graph. Lastly, 'Similar Articles' provides a list of articles with diverse viewpoints, each accompanied by a QR code, header, and a brief summary. "
        if instruction_type == 'ADD':
            task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element will be added to within the existing infographic in response to '{instruction}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
        elif instruction_type == 'DELETE':
            task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element will be deleted from within the existing infographic in response to '{instruction}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
        elif instruction_type == 'EDIT':
            task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element will be edited or modified within the existing infographic in response to '{instruction}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
        else:
            task = f"Based on the given description of an infographic with various sections (Header, Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles), infer the section in which the target element is originally in within the existing infographic in response to '{instruction}'. Provide one of the following answers: Number of Shares, Vote on Reliability, Related Facts, Latest Comments, Knowledge Graph Summaries, Similar Articles, Header, or NONE if unable to infer the infographic section."
        prompt = infographic_sections + task

        prediction_obj = generate_text(prompt)
        prediction = prediction_obj[0]['generated_text']
        predictions.append(prediction)
        references.append(infographic_section)

    prediction_to_int_mapping = {label: integer for integer, label in enumerate(set(predictions))}
    predictions = list(map(lambda x: prediction_to_int_mapping[x], predictions))
    references_to_int_mapping = {label: integer for integer, label in enumerate(set(references))}
    references = list(map(lambda x: references_to_int_mapping[x], references))
    accuracy_metric = evaluate.load("accuracy")
    results = accuracy_metric.compute(references=references, predictions=predictions)
    print('--------------EVALUATION RESULTS (INFOGRAPHIC SECTION)------------------')
    print(results)
    print('--------------EVALUATION RESULTS (INFOGRAPHIC SECTION)------------------')

def fine_tune_target_element_lora(input_model_path, output_model_path):
    input_column = 'instruction'
    output_column = 'target_element'
    instruction_type_column = 'instruction_type'

    with CodeTimer('Load Dataset', unit='s'):
        disable_caching()
        dataset = load_dataset("McSpicyWithMilo/target-elements-0.2split")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    with CodeTimer('Load Model', unit='s'):
        model = AutoModelForSeq2SeqLM.from_pretrained(input_model_path)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)

    with CodeTimer('Load Tokenizer', unit='s'):
        tokenizer = AutoTokenizer.from_pretrained(input_model_path)

    prefix = 'answer the question: '

    def pre_processing_function(sample):
        instruction_type = sample[instruction_type_column]
        if instruction_type == 'ADD':
            prompt_template = f'Given the instruction, ({{input}}) Output the specific content or element that is added to the infographic. Ensure that the answer does not include the target location of the element or text.'
        elif instruction_type == 'DELETE':
            prompt_template = f'Given the instruction, ({{input}}) Output the specific content or element that is deleted from the infographic. Ensure that the answer does not include the target location of the element or text.'
        elif instruction_type == 'EDIT':
            prompt_template = f'Given the instruction, ({{input}}) Output the specific content or element that is modified in the infographic. Ensure that the answer does not include the target location of the element or text.'
        else:
            prompt_template = f'Given the instruction, ({{input}}) Output the specific content or element that is moved within the infographic. Ensure that the answer does not include the target location of the element or text.'
        inputs = [prefix + prompt_template.format(input=item) for item in sample[input_column]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        labels = tokenizer(text_target=sample[output_column], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = train_dataset.map(pre_processing_function, batched=True)
    tokenized_test_dataset = test_dataset.map(pre_processing_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    with CodeTimer('Load Trainer', unit='s'):
        output_dir = './results'
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            auto_find_batch_size=True,
            learning_rate=1e-3,
            num_train_epochs=10,
            weight_decay=0.01,
            save_strategy="no",
            evaluation_strategy="epoch",
            report_to="tensorboard",
        )

        print('-------------TRAINING HYPER-PARAMETERS----------------')
        print('learning_rate = 1e-3')
        print('num_train_epochs = 10')
        print('weight_decay = 0.01')
        print('-------------TRAINING HYPER-PARAMETERS----------------')

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

    with CodeTimer('Evaluation Of The Model (Using Trainer)', unit='s'):
        results = trainer.evaluate()
        print(results)

    with CodeTimer('Saving Model and Tokenizer', unit='s'):
        trainer.model.save_pretrained(output_model_path)
        trainer.model.base_model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_model_path)

    with CodeTimer('Evaluation Of The Model (BLEU, ROUGE Metrics)', unit='s'):
        evaluate_target_element_lora(output_model_path)

def evaluate_target_element_lora(model_path):
    disable_caching()
    eval_dataset = load_dataset("McSpicyWithMilo/target-elements-0.2split", split="test")
    predictions, references = [], []
    generate_text = generate_local_pipeline(model_path)

    for row in eval_dataset:
        instruction = row["instruction"]
        instruction_type = row["instruction_type"]
        target_element = row["target_element"]

        if instruction_type == 'ADD':
            prompt = f'Given the instruction, ({instruction}) Output the specific content or element that is added to the infographic. Ensure that the answer does not include the target location of the element or text.'
        elif instruction_type == 'DELETE':
            prompt = f'Given the instruction, ({instruction}) Output the specific content or element that is deleted from the infographic. Ensure that the answer does not include the target location of the element or text.'
        elif instruction_type == 'EDIT':
            prompt = f'Given the instruction, ({instruction}) Output the specific content or element that is modified in the infographic. Ensure that the answer does not include the target location of the element or text.'
        else:
            prompt = f'Given the instruction, ({instruction}) Output the specific content or element that is moved within the infographic. Ensure that the answer does not include the target location of the element or text.'

        prediction_obj = generate_text(prompt)
        prediction = prediction_obj[0]['generated_text']
        predictions.append(prediction)
        references.append(target_element)

    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(references=references, predictions=predictions)
    rouge_metric = evaluate.load('rouge')
    rouge_results = rouge_metric.compute(references=references, predictions=predictions)
    print('--------------EVALUATION RESULTS (TARGET ELEMENT)------------------')
    print(bleu_results)
    print(rouge_results)
    print('--------------EVALUATION RESULTS (TARGET ELEMENT)------------------')

def fine_tune_target_location_lora(input_model_path, output_model_path):
    input_column = 'instruction'
    output_column = 'target_location'
    prompt_template = f'Identify the target location in the existing infographic where the new element should be placed based on the following user instruction: {{input}}'

    with CodeTimer('Load Dataset', unit='s'):
        disable_caching()
        dataset = load_dataset("McSpicyWithMilo/target-locations-0.2split")
        train_dataset = dataset["train"]
        test_dataset = dataset["test"]

    with CodeTimer('Load Model', unit='s'):
        model = AutoModelForSeq2SeqLM.from_pretrained(input_model_path)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )

        # add LoRA adaptor
        model = get_peft_model(model, lora_config)

    with CodeTimer('Load Tokenizer', unit='s'):
        tokenizer = AutoTokenizer.from_pretrained(input_model_path)

    prefix = 'answer the question: '

    def pre_processing_function(sample):
        inputs = [prefix + prompt_template.format(input=item) for item in sample[input_column]]
        model_inputs = tokenizer(inputs, max_length=128, truncation=True)

        labels = tokenizer(text_target=sample[output_column], max_length=512, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train_dataset = train_dataset.map(pre_processing_function, batched=True)
    tokenized_test_dataset = test_dataset.map(pre_processing_function, batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

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
            report_to="tensorboard",
        )

        print('-------------TRAINING HYPER-PARAMETERS----------------')
        print('learning_rate = 3e-4')
        print('num_train_epochs = 5')
        print('weight_decay = 0.01')
        print('-------------TRAINING HYPER-PARAMETERS----------------')

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

    with CodeTimer('Evaluation Of The Model (Using Trainer)', unit='s'):
        results = trainer.evaluate()
        print(results)

    with CodeTimer('Saving Model and Tokenizer', unit='s'):
        trainer.model.save_pretrained(output_model_path)
        trainer.model.base_model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_model_path)

    with CodeTimer('Evaluation Of The Model (BLEU, ROUGE Metrics)', unit='s'):
        evaluate_target_location_lora(output_model_path)

def evaluate_target_location_lora(model_path):
    disable_caching()
    eval_dataset = load_dataset("McSpicyWithMilo/target-locations-0.2split", split="test")
    predictions, references = [], []
    generate_text = generate_local_pipeline(model_path)

    for row in eval_dataset:
        instruction = row["instruction"]
        target_location = row["target_location"]

        prompt = f'Identify the target location in the existing infographic where the new element should be placed based on the following user instruction: {instruction}'

        prediction_obj = generate_text(prompt)
        prediction = prediction_obj[0]['generated_text']
        predictions.append(prediction)
        references.append(target_location)

    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(references=references, predictions=predictions)
    rouge_metric = evaluate.load('rouge')
    rouge_results = rouge_metric.compute(references=references, predictions=predictions)
    print('--------------EVALUATION RESULTS (TARGET LOCATION)------------------')
    print(bleu_results)
    print(rouge_results)
    print('--------------EVALUATION RESULTS (TARGET LOCATION)------------------')

if __name__ == '__main__':
    input_model_path = 'google/flan-t5-large'
    output_model_path = 'google/flan-t5-large-target-element-400-lora-tt20'
    # fine_tune_target_location_lora(input_model_path, output_model_path)
    evaluate_target_element_lora(output_model_path)
