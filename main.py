import json
import re
import os
from tqdm import tqdm
from configs import Configs
from generator import Generator
from retriever_bm25 import Retriever as RetrieverBM25
from test_case_runner import TestCaseRunner
from statistic import Statistic
from dataset import Dataset


def generate_all_test_cases(generator):
    dataset = Dataset(configs)
    samples = dataset.load_raw_data()
    test_cases = generator.generate_all_test_cases(samples)
    
    os.makedirs(os.path.dirname(configs.test_case_initial_gen_save_path), exist_ok=True)
    with open(configs.test_case_initial_gen_save_path, 'w') as f:
        json.dump(test_cases, f, indent=4)


def pipeline_for_generation_with_rag():
    # load target focal methods
    dataset = Dataset(configs)
    coverage_data = dataset.load_coverage_data_jacoco()

    # generating test case
    generated_test_cases = []  # list[(focal_file_path, generation_no_ref, generation_with_human_ref, generation_with_rag_ref)]
    generator = Generator(configs)

    for target_pair_idx, each_target_pair in tqdm(enumerate(coverage_data), total=len(coverage_data), ncols=80, desc='Generating test cases'):
        focal_file_path, target_focal_method, target_coverage, context, target_test_case, references_human = each_target_pair.focal_file_path, each_target_pair.focal_method, each_target_pair.coverage, each_target_pair.context, each_target_pair.test_case, each_target_pair.references

        # prepare retriever
        # prepare corpus. remove the target pair from the corpus
        coverage_data_for_corpus = coverage_data[:target_pair_idx] + coverage_data[target_pair_idx+1:]
        corpus_cov, corpus_fm, corpus_tc = [], [], []
        for each_pair_cor in coverage_data_for_corpus:
            corpus_cov.append(each_pair_cor.coverage)
            corpus_fm.append(each_pair_cor.focal_method)
            corpus_tc.append(each_pair_cor.test_case)
            
        retriever = RetrieverBM25(corpus_cov, corpus_fm, corpus_tc)
        references_cov_rag, references_fm_rag, references_tc_rag = retriever.retrieve(target_fm=target_focal_method, top_k=configs.retrieval_top_k, mode=configs.retrieval_mode)

        # generate test cases
        # with no reference
        generation_no_ref = generator.generate_test_case(target_coverage, context)

        # TODO: check the reference_human
        # with human reference
        generation_with_human_ref = None
        if references_human is not None:
            generation_with_human_ref = generator.generate_test_case(target_coverage, context, references_test_case=references_human[0], references_coverage=references_human[1])

        # with rag reference
        generation_with_rag_ref = generator.generate_test_case(target_coverage, context, references_test_case=references_tc_rag, references_coverage=references_cov_rag)

        generated_test_cases.append({
            'focal_file_path': focal_file_path, 
            'generation_no_ref': generation_no_ref, 
            'generation_with_human_ref': generation_with_human_ref, 
            'generation_with_rag_ref': generation_with_rag_ref,
            'target_test_case': target_test_case
            })
    
    os.makedirs(os.path.dirname(configs.test_case_initial_gen_save_path), exist_ok=True)
    with open(configs.test_case_initial_gen_save_path, 'w') as f:
        json.dump(generated_test_cases, f, indent=4)


def process_generated_test_cases():
    def _process(init_generation):
        result = re.findall(r'```java\n(.*?)```', init_generation, re.DOTALL)
        if len(result) == 0:
            result = re.findall(r'```\n(.*?)```', init_generation, re.DOTALL)
        if len(result) == 0:
            print('[WARNING] Abnormal generated test case:\n', init_generation, '\n\n')
            return None

        processed_test_case = None
        for each_code in result:
            if '@Test' in each_code:
                processed_test_case = each_code
                break
        if processed_test_case is None:
            print('[WARNING] Abnormal generated test case:\n', init_generation, '\n\n')
            return None
        
        # get the class name of the test case
        class_name = re.search(r'\sclass\s+(.+?)\s', processed_test_case)
        if class_name is not None:
            class_name = class_name.group(1)
            if 'Test' not in class_name:
                raise ValueError(f'Invalid class name in the generated test case:\n{processed_test_case}. Maybe need manually check the extraction.')
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
        focal_file_path = each_test_case['focal_file_path']
        focal_case_dir = focal_file_path[:focal_file_path.rfind('/')]
        # test_case_name = focal_file_path.split('/')[-1].split('.')[0]
        test_case_dir = focal_case_dir.replace('/main/', '/test/')

        test_case_no_ref, class_name_no_ref = _process(each_test_case['generation_no_ref'])
        if test_case_no_ref is None:
            print(f'[WARNING] Abnormal test case: {focal_file_path}') 
            continue

        test_case_with_rag_ref, class_name_with_rag_ref = _process(each_test_case['generation_with_rag_ref'])
        if test_case_with_rag_ref is None:
            print(f'[WARNING] Abnormal test case: {focal_file_path}') 
            continue

        test_case_with_huam_ref, class_name_with_human_ref = None, None
        if each_test_case['generation_with_human_ref'] is not None:
            test_case_with_huam_ref, class_name_with_human_ref = _process(each_test_case['generation_with_human_ref'])
            if test_case_with_huam_ref is None:
                print(f'[WARNING] Abnormal test case: {focal_file_path}') 
                continue

        test_case_no_ref_path = f'{test_case_dir}/{class_name_no_ref}.java'
        test_case_with_rag_ref_path = f'{test_case_dir}/{class_name_with_rag_ref}.java'
        test_case_with_huam_ref_path = f'{test_case_dir}/{class_name_with_human_ref}.java' if test_case_with_huam_ref is not None else None

        target_test_case = each_test_case['target_test_case']
        
        saved_test_cases.append({
            'focal_file_path': focal_file_path, 
            'generation_no_ref_path': test_case_no_ref_path, 
            'generation_no_ref': test_case_no_ref,
            'generation_with_human_ref_path': test_case_with_huam_ref_path,
            'generation_with_human_ref': test_case_with_huam_ref,
            'generation_with_rag_ref_path': test_case_with_rag_ref_path,
            'generation_with_rag_ref': test_case_with_rag_ref,
            'target_test_case': target_test_case
        })

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
    statistic.cal_bleu_for_saved_file()


def main():
    # generate all test cases without rag
    # generator = Generator(configs)
    # generate_all_test_cases(generator)

    # generate all test cases with rag (BM25)
    pipeline_for_generation_with_rag()
    
    # # process the generated test cases
    # process_generated_test_cases()

    # # run all test cases
    # test_case_runner = TestCaseRunner(configs)
    # run_all_test_cases(test_case_runner)
    
    # # statistics of test case execution
    # statistic = Statistic(configs)
    # get_statistics(statistic)


if __name__ == '__main__':
    configs = Configs()
    print(f'Configs:\n{configs.__dict__}\n\n')

    print(f"Processing {configs.project_name}...\n\n")
    
    os.makedirs(configs.test_case_run_log_dir, exist_ok=True)

    # TEST
    project_corpus_path = f'{configs.root_dir}/rag_tester/data/raw_data/{configs.project_name}_valid_pairs.json'
    #

    main()
    
