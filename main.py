import json
import re
import os
import argparse
import sys
from tqdm import tqdm
from configs import Configs
from generator import Generator
from retriever_bm25 import Retriever as RetrieverBM25
from test_case_runner import TestCaseRunner
from statistic import Statistic
from dataset import Dataset
from processing_tc import process_generated_test_cases


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
    generated_test_cases = []  # list[(focal_file_path, generation_no_ref, generation_human_ref, generation_rag_ref)]
    generator = Generator(configs)

    for target_pair_idx, each_target_pair in tqdm(enumerate(coverage_data), total=len(coverage_data), ncols=80, desc='Generating test cases'):
        project_name = each_target_pair.project_name
        focal_file_path = each_target_pair.focal_file_path
        focal_method_name = each_target_pair.focal_method_name
        target_focal_method = each_target_pair.focal_method
        target_coverage = each_target_pair.coverage
        context = each_target_pair.context
        target_test_case = each_target_pair.test_case
        references_human = each_target_pair.references

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
        generation_human_ref = None
        if references_human is not None:
            generation_human_ref = generator.generate_test_case(target_coverage, context, references_test_case=references_human[0], references_coverage=references_human[1])

        # with rag reference
        generation_rag_ref = generator.generate_test_case(target_coverage, context, references_test_case=references_tc_rag, references_coverage=references_cov_rag)

        generated_test_cases.append({
            'project_name': project_name,
            'focal_file_path': focal_file_path, 
            'focal_method_name': focal_method_name,
            'generation_no_ref': generation_no_ref, 
            'generation_human_ref': generation_human_ref, 
            'generation_rag_ref': generation_rag_ref,
            'target_coverage': target_coverage,
            'target_test_case': target_test_case
            })
    
    os.makedirs(os.path.dirname(configs.test_case_initial_gen_save_path), exist_ok=True)
    with open(configs.test_case_initial_gen_save_path, 'w') as f:
        json.dump(generated_test_cases, f, indent=4)


def run_all_test_cases(test_case_runner):
    # run the generated test cases
    with open(configs.test_case_save_path, 'r') as f:
        test_cases = json.load(f)

    test_case_runner.run_all_test_cases(test_cases)


def get_statistics(statistic):
    statistic.count_test_case_pass()
    statistic.cal_bleu_for_test_cases(is_pass=False, is_common=False)
    statistic.cal_bleu_for_test_cases(is_pass=True, is_common=False)
    statistic.cal_bleu_for_test_cases(is_pass=True, is_common=True)

    print('Coverage analysis...')
    print('- All coverages:')
    statistic.analyze_coverage(is_ref='no_ref', n_cover_line_threshold=1)
    statistic.analyze_coverage(is_ref='rag_ref', n_cover_line_threshold=1)
    statistic.analyze_coverage(is_ref='no_ref', n_cover_line_threshold=2)
    statistic.analyze_coverage(is_ref='rag_ref', n_cover_line_threshold=2)
    statistic.analyze_coverage(is_ref='no_ref', n_cover_line_threshold=3)
    statistic.analyze_coverage(is_ref='rag_ref', n_cover_line_threshold=3)

    print('\n- Common Coverages:')
    statistic.analyze_coverage(is_ref='no_ref', n_cover_line_threshold=1, is_common=True)
    statistic.analyze_coverage(is_ref='rag_ref', n_cover_line_threshold=1, is_common=True)
    statistic.analyze_coverage(is_ref='no_ref', n_cover_line_threshold=2, is_common=True)
    statistic.analyze_coverage(is_ref='rag_ref', n_cover_line_threshold=2, is_common=True)
    statistic.analyze_coverage(is_ref='no_ref', n_cover_line_threshold=3, is_common=True)
    statistic.analyze_coverage(is_ref='rag_ref', n_cover_line_threshold=3, is_common=True)


def main():
    # generate all test cases without rag
    # generator = Generator(configs)
    # generate_all_test_cases(generator)

    # generate all test cases with rag (BM25)
    pipeline_for_generation_with_rag()
    
    # process the generated test cases
    process_generated_test_cases(configs.test_case_initial_gen_save_path, configs.test_case_save_path)

    # run all test cases
    if os.path.exists(configs.test_case_run_log_dir):
        os.system(f'rm -r {configs.test_case_run_log_dir}')
    os.makedirs(configs.test_case_run_log_dir)
    
    test_case_runner = TestCaseRunner(configs)
    run_all_test_cases(test_case_runner)
    
    # statistics of test case execution
    statistic = Statistic(configs)
    get_statistics(statistic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str)
    args = parser.parse_args()

    configs = Configs(args.project_name)

    print(f'Configs:\n{configs.__dict__}\n\n')

    print(f"Processing {configs.project_name}...\n\n")

    main()