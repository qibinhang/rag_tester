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


def pipeline_for_generation_with_rag():
    # load target focal methods
    raw_samples = get_samples(configs.samples_path)
    samples = []
    for each_sample in raw_samples:
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
        
        samples.append((focal_file_path, target_focal_method, context, best_reference_test_case, best_reference_focal_method))
    
    # load corpus
    # TODO remove duplicated pairs
    with open(project_corpus_path, 'r') as f:
        corpus = json.load(f)
    
    corpus_fp, corpus_mn = [], []
    corpus_fm, corpus_tc = [], []
    for each_pair in corpus:
        each_file_path, method_name, test_case, focal_method = each_pair
        corpus_fp.append(each_file_path)
        corpus_mn.append(method_name)
        corpus_fm.append(focal_method)
        corpus_tc.append(test_case)

    # generating test case
    generated_test_cases = []  # list[(focal_file_path, generation_no_ref, generation_with_human_ref, generation_with_rag_ref)]
    generator = Generator(configs)

    for each_sample in tqdm(samples, ncols=80, desc='Generating test cases'):
        focal_file_path, target_focal_method, context, best_reference_test_case, best_reference_focal_method = each_sample

        # prepare retriever
        # remove the target focal method from the corpus
        target_focal_method_name = target_focal_method.split('(')[0].split()[-1]

        corpus_fm_clean, corpus_tc_clean = [], []
        clean_corpus = list(filter(lambda x: x[0] != target_focal_method_name, zip(corpus_mn, corpus_fm, corpus_tc)))
        for each in clean_corpus:
            corpus_fm_clean.append(each[1])
            corpus_tc_clean.append(each[2])
            
        retriever = RetrieverBM25(corpus_fm_clean, corpus_tc_clean)
        reference_focal_methods, reference_test_cases = retriever.retrieve(target_fm=target_focal_method, top_k=3)

        # generate test cases
        # with no reference
        generation_no_ref = generator.generate_test_case(target_focal_method, context)

        # with human reference
        generation_with_human_ref = generator.generate_test_case(target_focal_method, context, best_reference_test_case, best_reference_focal_method)

        # with rag reference
        generation_with_rag_ref = generator.generate_test_case(target_focal_method, context, reference_test_cases[0], reference_focal_methods[0])

        generated_test_cases.append((focal_file_path, generation_no_ref, generation_with_human_ref, generation_with_rag_ref))
    
    os.makedirs(os.path.dirname(configs.test_case_initial_gen_save_path), exist_ok=True)
    with open(configs.test_case_initial_gen_save_path, 'w') as f:
        json.dump(generated_test_cases, f, indent=4)


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
        
        if len(each_test_case) == 4:
            test_case_with_rag_ref, class_name_with_rag_ref = _process(each_test_case[3])
            if test_case_with_rag_ref is None:
                print(f'[WARNING] Abnormal test case: {focal_method_path}')
                continue
            test_case_with_rag_ref_path = f'{test_case_dir}/{class_name_with_rag_ref}.java'
            saved_test_cases.append((focal_method_path, test_case_no_ref_path, test_case_no_ref, test_case_with_ref_path, test_case_with_ref, test_case_with_rag_ref_path, test_case_with_rag_ref))
        else:
            saved_test_cases.append((focal_method_path, test_case_no_ref_path, test_case_no_ref, test_case_with_ref_path, test_case_with_ref))

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
    # generate all test cases without rag
    # generator = Generator(configs)
    # generate_all_test_cases(generator)

    # generate all test cases with rag (BM25)
    pipeline_for_generation_with_rag()
    
    # process the generated test cases
    process_generated_test_cases()

    # run all test cases
    test_case_runner = TestCaseRunner(configs)
    run_all_test_cases(test_case_runner)
    
    # # statistics of test case execution
    statistic = Statistic(configs)
    get_statistics(statistic)


if __name__ == '__main__':
    environment = ['charlie', 'cluster'][0]
    project_name = ['spark', 'HdrHistogram'][0]
    llm_name = ['llama_3', 'llama_3:70b'][0]
    retrieval_mode = ['fm', 'tc', 'both'][0]

    version = f'v0.8_mode_{retrieval_mode}'
    version_intro = 'Add the ability to generate test cases with the help of RAG (BM25). Retrieval mode is fm.'
    configs = Configs(project_name, environment, llm_name, version)
    
    configs.max_input_len = 7000
    configs.max_num_generated_tokens = 1024
    configs.top_p = 0.95
    configs.tempurature = 0.1
    configs.verbose = True

    configs.llm_name = llm_name
    configs.version = version
    configs.retrieval_mode = retrieval_mode
    configs.version_intro = version_intro

    print(f'Configs:\n{configs.__dict__}\n\n')

    print(f"Processing {project_name}...\n\n")
    
    os.makedirs(configs.test_case_run_log_dir, exist_ok=True)

    # TEST
    project_corpus_path = f'{configs.root_dir}/rag_tester/data/raw_data/{project_name}_valid_pairs.json'
    #

    main()
    