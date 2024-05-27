import json
import re
import os
from tqdm import tqdm
from configs import Configs
from generator import Generator
from retriever_bm25 import Retriever as RetrieverBM25
from test_case_runner import TestCaseRunner
from statistic import Statistic


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


def generate_all_test_cases(generator):
    samples = get_samples(configs.samples_path)
    test_cases = generator.generate_all_test_cases(samples)
    
    os.makedirs(os.path.dirname(configs.test_case_initial_gen_save_path), exist_ok=True)
    with open(configs.test_case_initial_gen_save_path, 'w') as f:
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


    with open(configs.test_case_initial_gen_save_path, 'r') as f:
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

    os.makedirs(os.path.dirname(configs.test_case_save_path), exist_ok=True)
    with open(configs.test_case_save_path, 'w') as f:
        json.dump(saved_test_cases, f, indent=4)


def run_all_test_cases(test_case_runner):
    # run the generated test cases
    with open(configs.test_case_save_path, 'r') as f:
        test_cases = json.load(f)

    test_case_runner.run_all_test_cases(test_cases)


def get_statistics(statistic):
    statistic.analyze_test_case_pass()


def main():
    # generate all test cases
    generator = Generator(configs)
    generate_all_test_cases(generator)
    
    # process the generated test cases
    process_generated_test_cases()

    # run all test cases
    test_case_runner = TestCaseRunner(configs)
    run_all_test_cases(test_case_runner)
    
    # statistics of test case execution
    statistic = Statistic(configs)
    get_statistics(statistic)


if __name__ == '__main__':
    environment = ['charlie', 'cluster'][0]
    project_name = ['spark', 'HdrHistogram'][0]
    llm_name = ['llama_3', 'llama_3:70b'][0]
    version = 'v0.7.2'
    version_intro = 'refactor the code by adding Generator class'
    configs = Configs(project_name, environment, llm_name, version)
    
    configs.version = version
    configs.version_intro = version_intro
    configs.llm_name = llm_name
    configs.max_input_len = 7000
    configs.max_num_generated_tokens = 1024
    configs.top_p = 0.95
    configs.tempurature = 0.1
    configs.verbose = True

    print(f'Configs:\n{configs.__dict__}\n\n')

    print(f"Processing {project_name}...\n\n")
    
    os.makedirs(configs.test_case_run_log_dir, exist_ok=True)

    main()
    