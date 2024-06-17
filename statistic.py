import os
import re
import json
import evaluate


class Statistic():
    def __init__(self, configs):
        self.configs = configs
        self.test_case_log_analysis = self.analyze_test_case_run_logs()

    def analyze_test_case_run_logs(self):
        with open(self.configs.test_case_log_and_coverage_save_path, 'r') as f:
            test_case_log_and_cov = json.load(f)

        test_case_log_analysis = []
        for each_tc_log_cov in test_case_log_and_cov:
            for is_ref in ('no_ref', 'rag_ref'):  # TODO: add 'human_ref'
                log_path = each_tc_log_cov[f'log_path_{is_ref}']
                if log_path is not None:
                    result_type = self._analyze_tc_run_log(log_path)
                    each_tc_log_cov[f'result_{is_ref}'] = result_type
                else:
                    each_tc_log_cov[f'result_{is_ref}'] = None

            test_case_log_analysis.append(each_tc_log_cov)
        return test_case_log_analysis

    def _analyze_tc_run_log(self, log_path):
        # return: 'SUCCESS', 'FAIL_COMPILE', 'FAIL_EXECUTE', 'UNKNOWN'
        with open(log_path, 'r') as f:
            running_log = f.read()
        log_name = os.path.basename(log_path)

        # execution success
        test_run_info = re.search(r'Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)', running_log)
        if test_run_info is not None:
            test_run_info = test_run_info.groups()

            if int(test_run_info[0]) > 1:
                print(f'[INFO] Multiple test methods in a single test case: {log_name}')

            success = int(test_run_info[0]) - int(test_run_info[1]) - int(test_run_info[2]) - int(test_run_info[3])
            if success > 0:
                return 'SUCCESS'
            else:
                return 'FAIL_EXECUTE'
        elif 'COMPILATION ERROR' in running_log:
            return 'FAIL_COMPILE'
        elif 'BUILD FAILURE' in running_log:
            return 'FAIL_EXECUTE'
        else:
            print(f'[WARNING] Unknown error type: {log_name}')
            return 'UNKNOWN'

    def count_test_case_pass(self):
        for is_ref in ('no_ref', 'rag_ref'):  # TODO: add 'human_ref'
            success_pass, fail_compile, fail_execute, unknown = self._count_test_case_pass(is_ref)
            print(f'[{is_ref}]\nFail Compile: {len(fail_compile)}, Fail Execute: {len(fail_execute)}, Unknown: {len(unknown)}, Success Pass: {len(success_pass)}\n')

    def _count_test_case_pass(self, is_ref):
        success_pass, fail_compile, fail_execute, unknown = [], [], [], []
        for each_tc_log_cov in self.test_case_log_analysis:
            result_type = each_tc_log_cov[f'result_{is_ref}']
            if result_type == 'SUCCESS':
                success_pass.append(each_tc_log_cov)
            elif result_type == 'FAIL_COMPILE':
                fail_compile.append(each_tc_log_cov)
            elif result_type == 'FAIL_EXECUTE':
                fail_execute.append(each_tc_log_cov)
            elif result_type == 'UNKNOWN':
                unknown.append(each_tc_log_cov)
            else:
                raise ValueError(f'Unknown result type: {result_type}')
        return success_pass, fail_compile, fail_execute, unknown
    
    def cal_bleu_for_test_cases(self, is_pass: bool=False, is_common: bool=False):
        test_case_no_ref_target_pairs, test_case_rag_ref_target_pairs = self.load_test_cases(is_pass=is_pass, is_common=is_common)

        print(f'\n\nBLEU-4 Analysis: is_pass={is_pass}, is_common={is_common}')
        print(f'no_ref: {len(test_case_no_ref_target_pairs)}, rag_ref: {len(test_case_rag_ref_target_pairs)}')

        bleu_no_ref = self.cal_bleu(test_case_no_ref_target_pairs)
        print(f'BLEU-4 without reference: {bleu_no_ref:.4f}')

        bleu_with_rag_ref = self.cal_bleu(test_case_rag_ref_target_pairs)
        print(f'BLEU-4 with RAG reference: {bleu_with_rag_ref:.4f}\n')

        return bleu_no_ref, bleu_with_rag_ref

    def load_test_cases(self, is_pass: bool=False, is_common: bool=False):
        if is_common:
            is_pass = True

        test_case_no_ref_target_pairs, test_case_rag_ref_target_pairs = [], []
        for each_test_case in self.test_case_log_analysis:
            if is_common:
                if each_test_case['result_no_ref'] != 'SUCCESS' or each_test_case['result_rag_ref'] != 'SUCCESS':
                    continue
            
            target_test_case = each_test_case['target_test_case']
            test_case_no_ref = each_test_case['generation_no_ref']
            test_case_rag_ref = each_test_case['generation_rag_ref']
            
            if (not is_pass) or each_test_case['result_no_ref'] == 'SUCCESS':
                test_case_no_ref_target_pairs.append((test_case_no_ref, target_test_case))

            if (not is_pass) or each_test_case['result_rag_ref'] == 'SUCCESS':
                test_case_rag_ref_target_pairs.append((test_case_rag_ref, target_test_case))
            
        return test_case_no_ref_target_pairs, test_case_rag_ref_target_pairs

    def get_negative_rag_ref(self):
        print('\n\nNegative RAG Reference: ')

        for each_tc_log_cov in self.test_case_log_analysis:
            if each_tc_log_cov['result_no_ref'] == 'SUCCESS' and each_tc_log_cov['result_rag_ref'] != 'SUCCESS':
                print(f'- no_ref:\n{each_tc_log_cov["generation_no_ref"]}')
                print(f'- rag_ref:\n{each_tc_log_cov["generation_rag_ref"]}\n')
                print(f'- rag_references:\n{each_tc_log_cov["target_test_case"]}\n\n')

    def cal_bleu(self, generated_target_pairs):
        generated_test_cases, target_test_cases = [], []
        for each_pair in generated_target_pairs:
            generated_test_cases.append(each_pair[0])
            target_test_cases.append(each_pair[1])

        bleu = evaluate.load('bleu')
        bleu_score = bleu.compute(predictions=generated_test_cases, references=target_test_cases)
        bleu_4 = bleu_score['precisions'][3]

        return bleu_4
    
    def load_coverage(self, is_ref, n_cover_line_threshold: int=1):
        assert is_ref in ['no_ref', 'rag_ref']
        assert n_cover_line_threshold > 0
        
        coverages = []
        for each_tc_cov in self.test_case_log_analysis:
            if each_tc_cov[f'coverage_{is_ref}'] is None:
                continue
            
            target_coverage = each_tc_cov['target_coverage']
            if target_coverage.count('<COVER>') < n_cover_line_threshold:
                continue
            
            coverages.append(each_tc_cov)
        return coverages
                
    def analyze_coverage(self, is_ref, n_cover_line_threshold: int=1, is_common: bool=False):
    # Args:
    #     is_ref: str, ['no_ref', 'human_ref', 'rag_ref']
    #     n_cover_line_threshold: int, the minimum number of covered lines in the target coverage.
    #     is_common: bool, whether to analyze the common coverage.
        assert is_ref in ['no_ref', 'rag_ref']
        assert n_cover_line_threshold > 0
        
        coverages = self.load_coverage(is_ref, n_cover_line_threshold)

        if is_common:
            another_is_ref = 'rag_ref' if is_ref == 'no_ref' else 'no_ref'
            common_coverages = []
            for each_cov in coverages:
                if each_cov[f'coverage_{another_is_ref}'] is not None:
                    common_coverages.append(each_cov)
            coverages = common_coverages

        print(f'\n\nCoverage Analysis: is_ref={is_ref}, Threshold={n_cover_line_threshold}, is_common={is_common}\n')  

        if len(coverages) == 0:
            print(f'No target coverages meet the requirements.')
            return
        
        is_ref_coverage = []
        for each_cov in coverages:
            target_coverage = each_cov['target_coverage']
            focal_file_coverage = each_cov[f'coverage_{is_ref}']
            focal_method_name = each_cov['focal_method_name']
            is_ref_coverage.append({'target_coverage': target_coverage, 'focal_file_coverage': focal_file_coverage, 'focal_method_name': focal_method_name})

        self._analyze_coverage(is_ref_coverage)
    
    def _analyze_coverage(self, coverages):
        exact_match_cases, fully_cover_cases, cover_ratio_list = [], [], []
        for cov_info in coverages:
            target_focal_method_coverage = cov_info['target_coverage']
            focal_file_coverage = cov_info['focal_file_coverage']
            focal_method_name = cov_info['focal_method_name']
            focal_method_name = focal_method_name.split('::::')[1]
            
            generated_focal_method_coverage = self.get_generated_focal_method_coverage(focal_file_coverage, target_focal_method_coverage, focal_method_name)

            is_exact_match, is_fully_cover, cover_ratio = self.eval_generated_coverage(generated_focal_method_coverage, target_focal_method_coverage)

            exact_match_cases.append(is_exact_match)
            fully_cover_cases.append(is_fully_cover)
            cover_ratio_list.append(cover_ratio)

        avg_exact_match = sum(exact_match_cases) / len(exact_match_cases)
        avg_cover = sum(fully_cover_cases) / len(fully_cover_cases)
        avg_cover_ratio = sum(cover_ratio_list) / len(cover_ratio_list)
        
        print(f'Exact Match: {avg_exact_match:.2%} ({sum(exact_match_cases)}/{len(exact_match_cases)})')
        print(f'Fully Cover: {avg_cover:.2%} ({sum(fully_cover_cases)}/{len(fully_cover_cases)})')
        print(f'Cover Ratio: {avg_cover_ratio:.2%}')
    
    def get_generated_focal_method_coverage(self, focal_file_coverage, target_coverage, focal_method_name):
        focal_file = focal_file_coverage.strip().replace('<COVER>', '')
        focal_method = target_coverage.strip().replace('<COVER>', '')
        focal_file_lines = focal_file.split('\n')
        focal_method_lines = focal_method.split('\n')

        possible_fm_start_indices = []
        for ff_line_idx in range(len(focal_file_lines)):
            if focal_file_lines[ff_line_idx].strip() == focal_method_lines[0].strip() and focal_file_lines[ff_line_idx+1].strip() == focal_method_lines[1].strip():
                possible_fm_start_indices.append(ff_line_idx)

        if len(possible_fm_start_indices) != 1:
            print(f'focal_method_name: {focal_method_name}')
            print(f'focal_file_coverage: {focal_file_coverage}')
            print(f'possible_fm_start_indices: {possible_fm_start_indices}')
            print(f'target_coverage: {target_coverage}')
            raise ValueError(f'len(possible_fm_start_indices) != 1. {len(possible_fm_start_indices)}')
        
        possible_gen_fm_cov = focal_file_coverage.split('\n')[possible_fm_start_indices[0]: possible_fm_start_indices[0]+len(focal_method_lines)]
        possible_gen_fm_cov = '\n'.join(possible_gen_fm_cov)
        
        # Check again
        possible_fm_lines = possible_gen_fm_cov.replace('<COVER>', '').split('\n')
        for line_idx in range(len(possible_fm_lines)):
            if possible_fm_lines[line_idx].strip() != focal_method_lines[line_idx].strip():
                raise ValueError(f'possible_fm != focal_method.\nPossible_fm:\n{possible_fm_lines}\n\n\nFocal_method:\n{focal_method_lines}\n')

        return possible_gen_fm_cov
    
    def eval_generated_coverage(self, generated_focal_method_coverage, target_focal_method_coverage):
        covered_lines_generated = []
        for line_idx, line in enumerate(generated_focal_method_coverage.split('\n')):
            if '<COVER>' in line:
                covered_lines_generated.append(line_idx)

        covered_lines_target = []
        for line_idx, line in enumerate(target_focal_method_coverage.split('\n')):
            if '<COVER>' in line:
                covered_lines_target.append(line_idx)

        is_exact_match = 1 if covered_lines_generated == covered_lines_target else 0

        covered_lines_set_generated, covered_lines_set_target = set(covered_lines_generated), set(covered_lines_target)
        is_fully_cover = 1 if covered_lines_set_generated >= covered_lines_set_target else 0

        cover_ratio = len(covered_lines_set_generated & covered_lines_set_target) / len(covered_lines_set_target)

        return is_exact_match, is_fully_cover, cover_ratio