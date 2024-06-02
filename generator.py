import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from instruction_constructor import InstructionConstructor


class Generator():
    def __init__(self, configs):
        self.configs = configs
        self.tokenizer, self.model = self.prepare_llm(configs.llm_name)
        self.get_instruction = InstructionConstructor().instruct_for_test_case_generate_given_fm

    def prepare_llm(self, llm_name):
        if llm_name == 'llama_3':
            tokenizer, model = self.load_llama_3(size='8b')
        elif llm_name == 'llama_3:70b':
            tokenizer, model = self.load_llama_3(size='70b')
        else:
            raise ValueError('Invalid LLM name')
        return tokenizer, model

    def load_llama_3(self, size):
        access_token = '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'
        if size == '8b':
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        elif size == '70b':
            model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        else:
            raise ValueError('Invalid LLM size')

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=access_token,
        )
        return tokenizer, model

    def finetune_llm(self, train_data, valid_data, output_dir):
        pass

    def generate_test_case_using_llama3(self, target_focal_method, context, reference_test_case=None, reference_focal_method=None):
        context_tokens = self.tokenizer.encode(context)
        context = self.tokenizer.decode(context_tokens[:self.configs.max_context_len], skip_special_tokens=True)
        messages = self.get_instruction(target_focal_method, context, reference_test_case, reference_focal_method)

        if self.configs.verbose:
            print('\n\n## System message ##')
            print(messages[0]['content'])
            print('\n\n## User message ##')
            print(messages[1]['content'])
            print('\n\n')

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=8000
        ).to(self.model.device)

        if self.configs.verbose:
            print(f'Length of input: {input_ids.shape[-1]}')

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.model.generate(
            input_ids,
            max_new_tokens=self.configs.max_num_generated_tokens,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=self.configs.tempurature,
            top_p=self.configs.top_p,
        )
        response = outputs[0][input_ids.shape[-1]:]
        generated_test_case = self.tokenizer.decode(response, skip_special_tokens=True)

        return generated_test_case
    
    def generate_test_case(self, target_focal_method, context, reference_test_case=None, reference_focal_method=None):
        if self.configs.llm_name in ('llama_3','llama_3:70b'):
            generation = self.generate_test_case_using_llama3(target_focal_method, context, reference_test_case, reference_focal_method)
        else:
            raise ValueError('Invalid LLM name')
        
        if self.configs.verbose:
            if reference_test_case is not None:
                print(f'\n\n## Generated test case WITH reference ##')
            else:
                print(f'\n\n## Generated test case WITHOUT reference ##')
            print(generation)
            print('\n\n')

        return generation

    def generate_all_test_cases(self, samples):
        test_cases = []

        # generate test cases for each sample
        for each_sample in tqdm(samples, ncols=80, desc='Generating test cases'):
            if self.configs.verbose:
                print('\n\n', '='*100)
                print(f'## Processing {each_sample[0]} ##')

            # each sample: (focal_file_path, target_focal_method, target_test_case, references)
            focal_file_path = each_sample[0]
            target_focal_method, reference_test_cases = each_sample[1], each_sample[3]
            best_reference_focal_method = reference_test_cases[0][0]
            best_reference_test_case = reference_test_cases[0][1]
            
            with open(focal_file_path, 'r') as f:
                context = f.read()

            # remove the copyright.
            context_lines = context.split('\n')
            for i, line in enumerate(context_lines):
                line = line.strip()
                if line.startswith('package') and line.endswith(';'):
                    context = '\n'.join(context_lines[i:])
                    break
            
            test_case_no_ref = self.generate_test_case(target_focal_method, context)
            test_case_with_ref = self.generate_test_case(target_focal_method, context, best_reference_test_case, best_reference_focal_method)
            
            test_cases.append((focal_file_path, test_case_no_ref, test_case_with_ref))

        return test_cases