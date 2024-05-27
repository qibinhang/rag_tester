import json
import torch    
import re
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs import Configs
from retriever_bm25 import RetrieverBM25


def get_samples(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    # load the data from the json file
    samples = []
    for focal_file_path, focal_methods in data.items():
        for each_focal_method in focal_methods:
            target_focal_method = each_focal_method['target_focal_method']
            target_test_case = each_focal_method['target_test_case']
            references = each_focal_method['references']
            samples.append((focal_file_path, target_focal_method, target_test_case, references))
    return samples


def load_llama_3(size):
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


def prepare_llm(llm_name):
    if llm_name == 'llama_3':
        tokenizer, model = load_llama_3(size='8b')
    elif llm_name == 'llama_3:70b':
        tokenizer, model = load_llama_3(size='70b')
    else:
        raise ValueError('Invalid LLM name')
    return tokenizer, model


def generate_test_case(target_focal_method, context, reference_test_case=None, reference_focal_method=None):
    if llm_name == 'llama_3' or llm_name == 'llama_3:70b':
        generation = generate_test_cases_using_llama3(target_focal_method, context, reference_test_case, reference_focal_method)
    else:
        raise ValueError('Invalid LLM name')
    
    if verbose:
        if reference_test_case is not None:
            print(f'\n\n## Generated test case WITH reference ##')
        else:
            print(f'\n\n## Generated test case WITHOUT reference ##')
        print(generation)
        print('\n\n')

    return generation


def construct_prompt(target_focal_method, context, reference_test_case=None, reference_focal_method=None):
    system_message = [{"role": "system", "content": "You are an expert in Junit test case generation. I will give you a target focal method, then you need to generate a JUnit test case with Junit version=4.12 and Java version=1.8. The generated test case must contain one test class and one test method and should be runnable. You must think carefully and pay attention to syntactic correctness.\nThe following is a Junit test case as an example, which contains test class RouteImplTest and test method testGets_thenReturnGetPathAndGetAcceptTypeSuccessfully().\n```java\npackage spark;\n\nimport org.junit.Test;\n\nimport static junit.framework.TestCase.assertNull;\nimport static org.junit.Assert.assertEquals;\nimport static org.junit.Assert.assertNotNull;\n\npublic class RouteImplTest {\n\n    private final static String PATH_TEST = \"/opt/test\";\n    private final static String ACCEPT_TYPE_TEST  = \"*/test\";\n\n    private RouteImpl route;\n\n    @Test\n    public void testGets_thenReturnGetPathAndGetAcceptTypeSuccessfully() throws Exception {\n        route = RouteImpl.create(PATH_TEST, ACCEPT_TYPE_TEST, null);\n        assertEquals(\"Should return path specified\", PATH_TEST, route.getPath());\n        assertEquals(\"Should return accept type specified\", ACCEPT_TYPE_TEST, route.getAcceptType());\n    }\n}```"}]

    user_content = f"The following is the target focal method that you need to generate a test case. Remember, the generated test case must contain one test class and one test method.\n```\n{target_focal_method}\n```"

    if reference_test_case is not None:
        assert reference_focal_method is not None
        user_content += f'\n\nThe following is a reference test case that might be referenced:\n```\n{reference_test_case}\n```'
        # user_content += f'\n\nThe following is a reference test case that might be referenced:\n```\n{reference_test_case}\n```\nThe reference test case is used to test the following reference focal method:\n```\n{reference_focal_method}\n```\n'

    user_content += f'\n\nThe following is the java file that the target focal method belongs to:\n```\n{context}\n```'

    messages = system_message + [{"role": "user", "content": user_content}]

    if verbose:
        print('\n\n## User message ##')
        print(user_content)
        print('\n\n')

    return messages


def generate_test_cases_using_llama3(target_focal_method, context, reference_test_case=None, reference_focal_method=None):
    messages = construct_prompt(target_focal_method, context, reference_test_case, reference_focal_method)

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_num_generated_tokens,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=tempurature,
        top_p=top_p,
    )
    response = outputs[0][input_ids.shape[-1]:]
    generated_test_case = tokenizer.decode(response, skip_special_tokens=True)

    return generated_test_case


def generate_all_test_cases():
    samples = get_samples(configs.samples_path)
    test_cases = []

    # generate test cases for each sample
    for each_sample in tqdm(samples, ncols=80, desc='Generating test cases'):
        if verbose:
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
        
        test_case_no_ref = generate_test_case(target_focal_method, context)
        test_case_with_ref = generate_test_case(target_focal_method, context, best_reference_test_case, best_reference_focal_method)
        
        test_cases.append((focal_file_path, test_case_no_ref, test_case_with_ref))
    
    os.makedirs(os.path.dirname(test_case_initial_gen_save_path), exist_ok=True)
    with open(test_case_initial_gen_save_path, 'w') as f:
        json.dump(test_cases, f, indent=4)


def process_generated_test_cases():
    def _process(init_generation):
        result = re.search(r'```java\n(.*)```', init_generation, re.DOTALL)
        if result is None:
            result = re.search(r'```(.*)```', init_generation, re.DOTALL)
        try:
            processed_test_case = result.group(1)
            assert len(processed_test_case) > 0
        except:
            print('[WARNING] Abnormal generated test case:\n', init_generation, '\n\n')
            return None
        
        # get the class name of the test case
        class_name = re.search(r'\sclass\s+(.+?)\s', processed_test_case)
        if class_name is not None:
            class_name = class_name.group(1)
            if 'Test' not in class_name:
                raise ValueError(f'Invalid class name in the generated test case:\n{processed_test_case}')
        else:
            print('[WARNING] Cannot find the class name in the generated test case:\n', processed_test_case, '\n\n')
            # find the method name
            class_name = re.search(r'\s(.+?)\s*\(', processed_test_case)
            if class_name is not None:
                class_name = class_name.group(1)
            else:
                raise('[WARNING] Cannot find the method name in the generated test case:\n', processed_test_case, '\n\n')

        return processed_test_case, class_name


    with open(test_case_initial_gen_save_path, 'r') as f:
        test_cases = json.load(f)

    # save the generated test cases
    saved_test_cases = []  # will be saved to a json file
    for each_test_case in tqdm(test_cases, ncols=80, desc='Processing generated test cases'):
        focal_method_path = each_test_case[0]
        focal_case_dir = focal_method_path[:focal_method_path.rfind('/')]
        # test_case_name = focal_method_path.split('/')[-1].split('.')[0]
        test_case_dir = focal_case_dir.replace('/main/', '/test/')

        test_case_no_ref, class_name_no_ref = _process(each_test_case[1])
        test_case_with_ref, class_name_with_ref = _process(each_test_case[2])
        if test_case_no_ref is None or test_case_with_ref is None:
            print(f'[WARNING] Abnormal test case: {focal_method_path}') 
            continue

        test_case_no_ref_path = f'{test_case_dir}/{class_name_no_ref}.java'
        test_case_with_ref_path = f'{test_case_dir}/{class_name_with_ref}.java'
        
        saved_test_cases.append((test_case_no_ref_path, test_case_no_ref, test_case_with_ref_path, test_case_with_ref, focal_method_path))

    os.makedirs(os.path.dirname(test_case_save_path), exist_ok=True)
    with open(test_case_save_path, 'w') as f:
        json.dump(saved_test_cases, f, indent=4)


def run_all_test_cases():
    # run the generated test cases
    with open(test_case_save_path, 'r') as f:
        test_cases = json.load(f)

    for each_test_case_path in tqdm(test_cases, ncols=80, desc='Running test cases'):
        test_case_no_ref_path, test_case_no_ref, test_case_with_ref_path, test_case_with_ref, focal_method_path = each_test_case_path

        print('Running the test case without reference...')
        os.makedirs(os.path.dirname(test_case_no_ref_path), exist_ok=True)
        with open(test_case_no_ref_path, 'w') as f:
            f.write(test_case_no_ref)

        run_test_case(test_case_no_ref_path, focal_method_path, is_ref=False)
        os.remove(test_case_no_ref_path)
        
        print('Running the test case with reference...')
        os.makedirs(os.path.dirname(test_case_with_ref_path), exist_ok=True)
        with open(test_case_with_ref_path, 'w') as f:
            f.write(test_case_with_ref)
        
        run_test_case(test_case_with_ref_path, focal_method_path, is_ref=True)
        os.remove(test_case_with_ref_path)


def run_test_case(test_case_path, focal_method_path, is_ref):
    test_case_relative_path = test_case_path.replace(configs.project_test_case_base_path, '')
    test_case_relative_path = test_case_relative_path.replace('.java', '')
    test_case_relative_path = test_case_relative_path.replace('/', '.')

    focal_method_name = focal_method_path.split('/')[-1].split('.')[0]

    suffix = 'with_ref' if is_ref else 'no_ref'
    index = 1
    log_file_path = f'{test_case_run_log_dir}/{focal_method_name}_{suffix}_{index}.log'
    while os.path.exists(log_file_path):
        index += 1
        log_file_path = f'{test_case_run_log_dir}/{focal_method_name}_{suffix}_{index}.log'

    cmd = f'cd {configs.project_dir} && mvn clean verify -Dtest={test_case_relative_path} > {log_file_path} 2>&1'

    print(cmd)
    os.system(cmd)


def statistics():
    def _analyze_log(log_names):
        fail_compile, fail_execute, success_pass = [], [], []
        fail_compile_count, fail_execute_count, success_pass_count = 0, 0, 0
        for each_log in log_names:
            running_log_path = os.path.join(test_case_run_log_dir, each_log)
            with open(running_log_path, 'r') as f:
                running_log = f.read()

            # execution success
            test_run_info = re.search(r'Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)', running_log)
            if test_run_info is not None:
                test_run_info = test_run_info.groups()

                if int(test_run_info[0]) > 1:
                    print(f'[INFO] Multiple test methods in a single test case: {each_log}')

                success = int(test_run_info[0]) - int(test_run_info[1]) - int(test_run_info[2]) - int(test_run_info[3])
                if success > 0:
                    success_pass.append(each_log)
                    success_pass_count += success

                if success != int(test_run_info[0]):
                    fail_execute.append(each_log)
                    failures_errors = int(test_run_info[1]) + int(test_run_info[2]) + int(test_run_info[3])
                    fail_execute_count += failures_errors
            elif 'COMPILATION ERROR' in running_log:
                fail_compile.append(each_log)
                fail_compile_count += 1
            elif 'BUILD FAILURE' in running_log:
                fail_execute.append(each_log)
                fail_execute_count += 1
            else:
                print(f'[WARNING] Unknown error type: {running_log_path}')

        return fail_compile, fail_execute, success_pass, fail_compile_count, fail_execute_count, success_pass_count
    
    no_ref_log_names, with_ref_log_names = [], []
    for each_log in os.listdir(test_case_run_log_dir):
        if 'no_ref' in each_log:
            no_ref_log_names.append(each_log)
        else:
            with_ref_log_names.append(each_log)
    assert len(no_ref_log_names) == len(with_ref_log_names)

    no_ref_fail_compile, no_ref_fail_execute, no_ref_success_pass, no_ref_fail_compile_count, no_ref_fail_execute_count, no_ref_success_pass_count = _analyze_log(no_ref_log_names)
    with_ref_fail_compile, with_ref_fail_execute, with_ref_success_pass, with_ref_fail_compile_count, with_ref_fail_execute_count, with_ref_success_pass_count = _analyze_log(with_ref_log_names)

    print(f'[No Reference]\nFail Compile: {no_ref_fail_compile_count}, Fail Execute: {no_ref_fail_execute_count}, Success Pass: {no_ref_success_pass_count}\n\n')
    print(f'[With Reference]\nFail Compile: {with_ref_fail_compile_count}, Fail Execute: {with_ref_fail_execute_count}, Success Pass: {with_ref_success_pass_count}\n\n')

    print('Detailed information:')
    print(f'[No Reference]\nFail Compile: {no_ref_fail_compile}\n\nFail Execute: {no_ref_fail_execute}\n\nSuccess Pass: {no_ref_success_pass}\n\n')
    print(f'[With Reference]\nFail Compile: {with_ref_fail_compile}\n\nFail Execute: {with_ref_fail_execute}\n\nSuccess Pass: {with_ref_success_pass}\n\n')


def main():
    # generate all test cases
    global model, tokenizer
    tokenizer, model = prepare_llm(llm_name)
    generate_all_test_cases()
    
    # process the generated test cases
    process_generated_test_cases()

    # run all test cases
    run_all_test_cases()
    
    # statistics of test case execution
    statistics()


if __name__ == '__main__':
    environment = ['charlie', 'cluster'][0]
    project_name = ['spark', 'HdrHistogram'][0]
    llm_name = ['llama_3', 'llama_3:70b'][0]
    version = 'v0.7.1'  # add reference focal method
    configs = Configs(project_name, environment)

    print(f"Processing {project_name}...\n\n")
    
    max_input_len = 7000
    max_num_generated_tokens = 1024
    top_p = 0.95
    tempurature = 0.1
    verbose = True

    test_case_initial_gen_save_path = f'./data/generated_test_cases/{project_name}_{llm_name}_init_gen_{version}.json'
    test_case_save_path = f'./data/generated_test_cases/{project_name}_{llm_name}_processed_{version}.json'

    test_case_run_log_dir = os.path.abspath(f'./data/generated_test_cases_run_log/{project_name}_{llm_name}_{version}')
    os.makedirs(test_case_run_log_dir, exist_ok=True)

    main()
    