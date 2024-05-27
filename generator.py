import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class Generator():
    def __init__(self, configs):
        self.configs = configs
        self.tokenizer, self.model = self.prepare_llm(configs.llm_name)

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

    def construct_prompt(self, target_focal_method, context, reference_test_case=None, reference_focal_method=None):
        system_message = [{"role": "system", "content": "You are an expert in Junit test case generation. I will give you a target focal method, then you need to generate a JUnit test case with Junit version=4.12 and Java version=1.8. The generated test case must contain one test class and one test method and should be runnable. You must think carefully and pay attention to syntactic correctness.\nThe following is a Junit test case as an example, which contains test class RouteImplTest and test method testGets_thenReturnGetPathAndGetAcceptTypeSuccessfully().\n```java\npackage spark;\n\nimport org.junit.Test;\n\nimport static junit.framework.TestCase.assertNull;\nimport static org.junit.Assert.assertEquals;\nimport static org.junit.Assert.assertNotNull;\n\npublic class RouteImplTest {\n\n    private final static String PATH_TEST = \"/opt/test\";\n    private final static String ACCEPT_TYPE_TEST  = \"*/test\";\n\n    private RouteImpl route;\n\n    @Test\n    public void testGets_thenReturnGetPathAndGetAcceptTypeSuccessfully() throws Exception {\n        route = RouteImpl.create(PATH_TEST, ACCEPT_TYPE_TEST, null);\n        assertEquals(\"Should return path specified\", PATH_TEST, route.getPath());\n        assertEquals(\"Should return accept type specified\", ACCEPT_TYPE_TEST, route.getAcceptType());\n    }\n}```"}]

        user_content = f"The following is the target focal method that you need to generate a test case. Remember, the generated test case must contain one test class and one test method.\n```\n{target_focal_method}\n```"

        if reference_test_case is not None:
            assert reference_focal_method is not None
            # user_content += f'\n\nThe following is a reference test case that might be referenced:\n```\n{reference_test_case}\n```'
            user_content += f'\n\nThe following is a reference test case that might be referenced:\n```\n{reference_test_case}\n```\nThe reference test case is used to test the following reference focal method:\n```\n{reference_focal_method}\n```\n'

        user_content += f'\n\nThe following is the java file that the target focal method belongs to:\n```\n{context}\n```'

        messages = system_message + [{"role": "user", "content": user_content}]

        if self.configs.verbose:
            print('\n\n## User message ##')
            print(user_content)
            print('\n\n')

        return messages
    
    def generate_test_case_using_llama3(self, target_focal_method, context, reference_test_case=None, reference_focal_method=None):
        messages = self.construct_prompt(target_focal_method, context, reference_test_case, reference_focal_method)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.configs.max_input_len
        ).to(self.model.device)

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