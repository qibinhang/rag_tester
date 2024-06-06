import logging
from dataclasses import dataclass, field
import os
import random
import torch
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
        set_seed,

)
from trl import setup_chat_format
from peft import LoraConfig
from trl import SFTTrainer

from dataset import Dataset as CoverageDataset
from instruction_constructor import InstructionConstructor
from configs import Configs as ProjectConfigs

# Comment in if you want to use the Llama 3 instruct template but make sure to add modules_to_save
# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)


@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=3072, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )

    access_token: str = field(
        default="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"
    )


def load_instruct_data(task: str):
    if task == 'cov_pred_given_tc':
        return load_fm_tc2cov_instruct_data()
    elif task == 'imp_cov_pred':
        raise NotImplementedError('Not implemented yet')
    elif task == 'tc_gen':
        raise NotImplementedError('Not implemented yet')
    else:
        raise ValueError('Invalid task')


def load_fm_tc2cov_instruct_data():
    configs = ProjectConfigs()
    dataset = CoverageDataset(configs)
    coverage_data = dataset.load_coverage_data(label_method='human') # format: [<Coverage, Context, Test case>]

    # convert format by adding instruction: 
    # list({"messages": [{"role":"system", "content": system_instruction,},{"role":"user", "content": user_instruction},{"role":"assistant", "content": assistant_instruction}]})
    fm_tc2cov_instruct_data = []
    instruction_constructor = InstructionConstructor()
    random.seed(training_args.seed)
    coverage_data_indices = list(range(len(coverage_data)))
    random.shuffle(coverage_data_indices)
    for i, each_coverage in enumerate(coverage_data):
        # add instructions of system and usesr
        coverage, context, test_case = each_coverage
        focal_method = coverage.replace("<COVER>", "")

        example_cov_context_tc = coverage_data[coverage_data_indices[i]]
        example_fm = example_cov_context_tc[0].replace("<COVER>", "")
        example_fm_context_tc_cov = [example_fm, example_cov_context_tc[1], example_cov_context_tc[2], example_cov_context_tc[0]]

        system_user_instruct = instruction_constructor.instruct_for_coverage_predict_given_tc(focal_method, context, test_case, example_fm_context_tc_cov)

        # add instruction of assistant
        assistant_instruct = {"role": "assistant", "content":f'\n```\n{coverage}```\n'}
        fm_tc2cov_instruct_data.append({"messages": system_user_instruct + [assistant_instruct]})
    return fm_tc2cov_instruct_data


def training_function(script_args, training_args):
    ################
    # Dataset
    ################

    instruct_data = load_instruct_data(task='cov_pred_given_tc')
    random.seed(training_args.seed)
    random.shuffle(instruct_data)
    train_dataset = instruct_data[:int(0.8 * len(instruct_data))]
    test_dataset = instruct_data[int(0.8 * len(instruct_data)):]
    train_dataset = datasets.Dataset.from_list(train_dataset)
    test_dataset = datasets.Dataset.from_list(test_dataset)

    ################
    # Model & Tokenizer
    ################

    # Tokenizer        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id, use_fast=True, token=script_args.access_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    tokenizer.add_special_tokens({'additional_special_tokens': ['<COVER>']})
    
    # template dataset
    def template_dataset(examples):
        return{"text":  tokenizer.apply_chat_template(examples["messages"], tokenize=False)}
    
    train_dataset = train_dataset.map(template_dataset, remove_columns=["messages"])
    test_dataset = test_dataset.map(template_dataset, remove_columns=["messages"])
    
    # print random sample
    with training_args.main_process_first(
        desc="Log a few random samples from the processed training set"
    ):
        for index in random.sample(range(len(train_dataset)), 1):
            print(train_dataset[index]["text"])

    # Model   
    model = AutoModelForCausalLM.from_pretrained(
            script_args.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=script_args.access_token,
            use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        )

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("INFO: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))
        
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save = ["embed_tokens"]
        # modules_to_save = ["lm_head", "embed_tokens"] # add if you want to use the Llama 3 instruct template
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    # trainer.save_model()
    print("Saving model to", training_args.output_dir)
    trainer.model.save_pretrained(training_args.output_dir, save_embedding_layers=True)
    

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # set seed
    set_seed(training_args.seed)
  
    # launch training
    training_function(script_args, training_args)