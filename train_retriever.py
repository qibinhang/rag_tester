from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from dataset import Dataset as CoverageDataset
from configs import Configs as ProjectConfigs
from datasets import Dataset


def load_coverage_data():
    configs = ProjectConfigs()
    dataset = CoverageDataset(configs)
    coverage_data = dataset.load_coverage_data_human_labeled() # format: [<Coverage, Context, Test case>]
    return coverage_data


def preprocess_dataset(dataset, tokenizer, max_source_length, max_target_length):
    model_inputs = []
    model_labels = []
    for each_pair in dataset:
        coverage, context, test_case = each_pair
        model_inputs.append(f'{coverage}\n{tokenizer.sep_token}\n{context}')
        model_labels.append(test_case)
    model_inputs = tokenizer(model_inputs, max_length=max_source_length, padding="max_length", truncation=True)
    model_labels = tokenizer(model_labels, max_length=max_target_length, padding="max_length", truncation=True).input_ids

    labels_with_ignore_index = []
    for each_label in model_labels:
        updated_label = [idx if idx != tokenizer.pad_token_id else -100 for idx in each_label]
        labels_with_ignore_index.append(updated_label)
    model_inputs['labels'] = labels_with_ignore_index
    dataset = Dataset.from_dict(model_inputs)
    dataset = dataset.train_test_split(test_size=0.2)
    return dataset['train'], dataset['test']


def main():
    model_name = configs.retriever_model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<COVER>']})

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    dataset = load_coverage_data()
    train_dataset, test_dataset = preprocess_dataset(dataset, tokenizer, max_source_length=configs.retriever_max_source_length, max_target_length=configs.retriever_max_target_length)

    # verify the dataset
    for batch in train_dataset:
        print(batch.keys())
        print(tokenizer.decode(batch['input_ids']))
        print(tokenizer.decode([label for label in batch['labels'] if label != -100]))
        break
    
    training_args = TrainingArguments(
    output_dir=configs.retriever_model_dir,
    do_train=True,
    # save_strategy="epoch",
    # save_total_limit=2,

    num_train_epochs=configs.retriever_n_epochs,
    evaluation_strategy="epoch",
    learning_rate=configs.retriever_lr,
    per_device_train_batch_size=configs.retriever_train_batch_size,
    per_device_eval_batch_size=configs.retriever_train_batch_size,
    gradient_accumulation_steps=configs.gradient_accumulation_steps,

    weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=0.2,

    dataloader_drop_last=True,
    dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    print(f"evaluation: {eval_results}")

    model.save_pretrained(f'{configs.retriever_model_dir}')
    tokenizer.save_pretrained(f'{configs.retriever_model_dir}')

    
if __name__ == "__main__":
    configs = ProjectConfigs()

    for key in dir(configs):
        if 'retriever' in key:
            print(key, getattr(configs, key))

    main()