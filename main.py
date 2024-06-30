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
from processing_tc import process_all_generated_test_cases
from test_case_refiner import TestCaseRefiner


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
    # print('Loading jacoco labelled coverages...')
    # coverage_data = dataset.load_coverage_data_jacoco()

    print('Loading focal methods...')
    coverage_data = dataset.load_focal_method_data()

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
        references_cov_rag, references_fm_rag, references_tc_rag, references_score = retriever.retrieve(target_fm=target_focal_method, top_k=configs.retrieval_top_k, mode=configs.retrieval_mode)

        # generate test cases
        # with no reference
        generation_no_ref = generator.generate_test_case(target_coverage, context, focal_method_name.split('::::')[0])

        # TODO: check the reference_human
        # with human reference
        generation_human_ref = None
        if references_human is not None:
            generation_human_ref = generator.generate_test_case(target_coverage, context, focal_method_name.split('::::')[0], references_test_case=references_human[0], references_coverage=references_human[1])

        # with rag reference
        generation_rag_ref = generator.generate_test_case(target_coverage, context, focal_method_name.split('::::')[0], references_test_case=references_tc_rag, references_coverage=references_cov_rag)
        rag_references = [(references_score[i], references_cov_rag[i], references_tc_rag[i]) for i in range(len(references_cov_rag))]

        generated_test_cases.append({
            'project_name': project_name,
            'target_coverage_idx': target_pair_idx,
            'focal_file_path': focal_file_path, 
            'focal_method_name': focal_method_name,
            'generation_no_ref': generation_no_ref, 
            'generation_human_ref': generation_human_ref, 
            'generation_rag_ref': generation_rag_ref,
            'rag_references': rag_references,
            'target_coverage': target_coverage,
            'target_context': context,
            'target_test_case': target_test_case
            })
    
    os.makedirs(os.path.dirname(configs.test_case_initial_gen_save_path), exist_ok=True)
    with open(configs.test_case_initial_gen_save_path, 'w') as f:
        json.dump(generated_test_cases, f, indent=4)


def run_all_test_cases(test_case_runner):
    # run the generated test cases
    with open(configs.test_case_save_path, 'r') as f:
        test_cases = json.load(f)

    no_ref_log_coverage = test_case_runner.run_all_test_cases(test_cases, is_ref='no_ref')
    rag_ref_log_coverage = test_case_runner.run_all_test_cases(test_cases, is_ref='rag_ref')

    # merge
    merge_log_coverage = no_ref_log_coverage
    for idx in range(len(no_ref_log_coverage)):
        assert no_ref_log_coverage[idx]['target_coverage_idx'] == rag_ref_log_coverage[idx]['target_coverage_idx']

        merge_log_coverage[idx]['log_path_rag_ref'] = rag_ref_log_coverage[idx]['log_path_rag_ref']
        merge_log_coverage[idx]['coverage_rag_ref'] = rag_ref_log_coverage[idx]['coverage_rag_ref']
            
    test_case_runner.save_log_coverage(merge_log_coverage, configs.test_case_log_and_coverage_save_path)


def get_statistics(statistic):
    statistic.count_test_case_pass(is_ref='no_ref')
    statistic.count_test_case_pass(is_ref='rag_ref')

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

    print('\n- In Project API analysis:')
    statistic.analyze_apis(configs.project_apis_extraction_save_path, f'{configs.project_dir}/{configs.project_name}', is_pass=False, is_common=False)
    statistic.analyze_apis(configs.project_apis_extraction_save_path, f'{configs.project_dir}/{configs.project_name}', is_pass=True, is_common=False)
    statistic.analyze_apis(configs.project_apis_extraction_save_path, f'{configs.project_dir}/{configs.project_name}', is_pass=True, is_common=True)

    statistic.analyze_positive_reg_ref_apis(configs.project_apis_extraction_save_path, f'{configs.project_dir}/{configs.project_name}')

    # analyze the positive and negative references
    # statistic.print_negative_rag_ref_pass()
    # statistic.print_positive_rag_ref_pass()
    # statistic.get_positive_negative_rag_ref_coverage()
    # statistic.get_negative_rag_ref_compilation()


def refine_test_case(generator, test_case_log_and_coverage_save_path, refined_test_case_save_dir):
    statistic = Statistic(test_case_log_and_coverage_save_path)
    tc_refiner = TestCaseRefiner(generator=generator)
    no_ref_refined_test_cases = tc_refiner.refine(
        test_case_log_and_coverage=statistic.test_case_log_analysis, 
        is_ref='no_ref'
        )
    tc_refiner.save_refined_test_cases(no_ref_refined_test_cases, f'{refined_test_case_save_dir}/no_ref.json')
    
    rag_ref_refined_test_cases = tc_refiner.refine(
        test_case_log_and_coverage=statistic.test_case_log_analysis, 
        is_ref='rag_ref'
        )
    tc_refiner.save_refined_test_cases(rag_ref_refined_test_cases, f'{refined_test_case_save_dir}/rag_ref.json')


def evaluate_refined_test_cases(refined_test_case_save_dir, is_ref):
    ## run the refined test cases
    with open(f'{refined_test_case_save_dir}/{is_ref}.json', 'r') as f:
        refined_test_cases = json.load(f)

    refined_test_case_run_log_dir = configs.get_refined_test_case_run_log_dir()

    test_case_runner = TestCaseRunner(
        configs, 
        test_case_run_log_dir=refined_test_case_run_log_dir
        )
    
    test_case_with_log_coverage = test_case_runner.run_all_test_cases(refined_test_cases, is_ref=is_ref)

    refined_log_coverage_save_dir = configs.get_refined_test_case_log_and_coverage_save_dir()
    refined_log_coverage_save_path = f'{refined_log_coverage_save_dir}/{is_ref}.json'

    test_case_runner.save_log_coverage(
        test_case_with_log_coverage, refined_log_coverage_save_path
        )

    ## statistics
    statistic = Statistic(refined_log_coverage_save_path)
    statistic.count_test_case_pass(is_ref=is_ref)


def merge_refined_test_cases(before_refine_test_case_log_cov_path, refined_test_case_log_cov_save_dir, save_path):
    with open(before_refine_test_case_log_cov_path, 'r') as f:
        before_refine_test_case_log_cov = json.load(f)
    
    for is_ref in ('no_ref', 'rag_ref'):
        with open(f'{refined_test_case_log_cov_save_dir}/{is_ref}.json', 'r') as f:
            refined_test_case_log_cov = json.load(f)

        for each_refined_test_case in refined_test_case_log_cov:
            refined_target_coverage_idx = each_refined_test_case['target_coverage_idx']
            for idx, each_before_refine_test_case in enumerate(before_refine_test_case_log_cov):
                if each_before_refine_test_case['target_coverage_idx'] == refined_target_coverage_idx:
                    before_refine_test_case_log_cov[idx][f'generation_{is_ref}'] = each_refined_test_case[f'generation_{is_ref}']
                    before_refine_test_case_log_cov[idx][f'generation_{is_ref}_path'] = each_refined_test_case[f'generation_{is_ref}_path']
                    before_refine_test_case_log_cov[idx][f'result_{is_ref}'] = each_refined_test_case[f'result_{is_ref}']
                    before_refine_test_case_log_cov[idx][f'log_path_{is_ref}'] = each_refined_test_case[f'log_path_{is_ref}']
                    before_refine_test_case_log_cov[idx][f'coverage_{is_ref}'] = each_refined_test_case[f'coverage_{is_ref}']

                    break
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(before_refine_test_case_log_cov, f, indent=4)


def main():
    # generate all test cases with rag (BM25)
    pipeline_for_generation_with_rag()
    
    # process the generated test cases
    process_all_generated_test_cases(configs.test_case_initial_gen_save_path, configs.test_case_save_path)

    # run all test cases
    if os.path.exists(configs.test_case_run_log_dir):
        os.system(f'rm -r {configs.test_case_run_log_dir}')
    os.makedirs(configs.test_case_run_log_dir)
    
    test_case_runner = TestCaseRunner(configs, configs.test_case_run_log_dir)
    run_all_test_cases(test_case_runner)
    
    # statistics of test case execution
    statistic = Statistic(configs.test_case_log_and_coverage_save_path)
    get_statistics(statistic)

    # refine round 1
    print('Refining test cases...')
    ## refine
    configs.refine_round = 1
    refined_test_case_save_dir = configs.get_refined_test_case_save_dir()

    generator = Generator(configs)
    refine_test_case(
        generator=generator,
        test_case_log_and_coverage_save_path=configs.test_case_log_and_coverage_save_path,
        refined_test_case_save_dir=refined_test_case_save_dir
        )
    
    ## evaluate
    refined_test_case_run_log_dir = configs.get_refined_test_case_run_log_dir()
    if os.path.exists(refined_test_case_run_log_dir):
        os.system(f'rm -r {refined_test_case_run_log_dir}')
    os.makedirs(refined_test_case_run_log_dir)

    evaluate_refined_test_cases(refined_test_case_save_dir, is_ref='no_ref')
    evaluate_refined_test_cases(refined_test_case_save_dir, is_ref='rag_ref')
    
    ## merge refined test case log and coverage results into final test case log and coverage results
    refined_log_coverage_save_dir = configs.get_refined_test_case_log_and_coverage_save_dir()

    final_tc_log_cov_save_path = configs.final_test_case_log_and_coverage_save_path
    before_refine_test_case_log_cov_path = configs.test_case_log_and_coverage_save_path

    merge_refined_test_cases(
        before_refine_test_case_log_cov_path, 
        refined_log_coverage_save_dir, 
        final_tc_log_cov_save_path
        )
    
    # statistics of final test cases coverages
    final_tc_log_cov_save_path = configs.final_test_case_log_and_coverage_save_path
    statistic = Statistic(final_tc_log_cov_save_path)
    get_statistics(statistic)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str)
    args = parser.parse_args()

    configs = Configs(args.project_name)

    print(f'Configs:\n{configs.__dict__}\n\n')

    print(f"Processing {configs.project_name}...\n\n")

    main()