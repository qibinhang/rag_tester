import os
import re
import json
import evaluate
from utils import load_project_apis, extract_method_invocation_from_java_code


class Statistic():
    def __init__(self, test_case_log_and_coverage_save_path):
        self.test_case_log_analysis = self.analyze_test_case_run_logs(test_case_log_and_coverage_save_path)

    def analyze_test_case_run_logs(self, test_case_log_and_coverage_save_path):
        with open(test_case_log_and_coverage_save_path, 'r') as f:
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

    def count_test_case_pass(self, is_ref):
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

    def load_test_cases(self, is_pass: bool=False, is_common: bool=False, return_focal_method=False):
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
            fm_name = each_test_case['focal_method_name']
            
            if (not is_pass) or each_test_case['result_no_ref'] == 'SUCCESS':
                no_ref_pair = [test_case_no_ref, target_test_case]
                if return_focal_method:
                    no_ref_pair.append(fm_name)
                test_case_no_ref_target_pairs.append(no_ref_pair)

            if (not is_pass) or each_test_case['result_rag_ref'] == 'SUCCESS':
                rag_ref_pair = [test_case_rag_ref, target_test_case]
                if return_focal_method:
                    rag_ref_pair.append(fm_name)
                test_case_rag_ref_target_pairs.append(rag_ref_pair)
            
        return test_case_no_ref_target_pairs, test_case_rag_ref_target_pairs

    def print_negative_rag_ref_pass(self):
        print('\n\nNegative RAG Reference: ')

        for each_tc_log_cov in self.test_case_log_analysis:
            if each_tc_log_cov['result_no_ref'] == 'SUCCESS' and each_tc_log_cov['result_rag_ref'] != 'SUCCESS':
                print(f'- no_ref:\n{each_tc_log_cov["generation_no_ref"]}')
                print(f'- rag_ref:\n{each_tc_log_cov["generation_rag_ref"]}\n')
                for each_ref in each_tc_log_cov['rag_references']:
                    print(f'- rag_ref score: {each_ref[0]}\n')
                    print(f'- rag_ref coverage:\n{each_ref[1]}\n')
                    print(f'- rag_ref test case:\n{each_ref[2]}\n')
                    print('-'*50)
                print('='*50)
    
    def print_positive_rag_ref_pass(self):
        print('\n\nPositive RAG Reference: ')
        pos_tc_log_cov = self.get_positive_rag_ref_pass()
        for each_tc_log_cov in pos_tc_log_cov:
            print(f'- focal_method:\n{each_tc_log_cov["target_coverage"]}')
            print(f'- no_ref:\n{each_tc_log_cov["generation_no_ref"]}')
            print(f'- rag_ref:\n{each_tc_log_cov["generation_rag_ref"]}\n')
            for each_ref in each_tc_log_cov['rag_references']:
                print(f'- rag_ref score: {each_ref[0]}\n')
                print(f'- rag_ref coverage:\n{each_ref[1]}\n')
                print(f'- rag_ref test case:\n{each_ref[2]}\n')
                print('-'*50)
            print('='*50)

    def get_positive_rag_ref_pass(self):
        pos_tc_log_cov = []
        for each_tc_log_cov in self.test_case_log_analysis:
            if each_tc_log_cov['result_no_ref'] != 'SUCCESS' and each_tc_log_cov['result_rag_ref'] == 'SUCCESS':
                pos_tc_log_cov.append(each_tc_log_cov)
        return pos_tc_log_cov

    def get_negative_rag_ref_compilation(self):
        compilation_error_type = ['cannot find symbol',]
        unknow_types = []
        negative_rag_ref_compilation = []
        each_error_type_count = {error_type: 0 for error_type in compilation_error_type}

        print('\n\nRAG Fail Compilation but NoRAG Success Compilation: ')
        for each_tc_log_cov in self.test_case_log_analysis:
            if each_tc_log_cov['result_no_ref'] != 'FAIL_COMPILE' and each_tc_log_cov['result_rag_ref'] == 'FAIL_COMPILE':
                negative_rag_ref_compilation.append(each_tc_log_cov)
        
        # count the error types
        for each_tc_log_cov in negative_rag_ref_compilation:
            log_path = each_tc_log_cov['log_path_rag_ref']
            with open(log_path, 'r') as f:
                running_log = f.readlines()
            
            is_unknow_type = True
            for error_type in compilation_error_type:
                for each_line in running_log:
                    if error_type in each_line:
                        each_error_type_count[error_type] += 1
                        is_unknow_type = False
                        break
            if is_unknow_type:
                unknow_types.append(log_path)

        print(f'Negative RAG Ref Compilation in Total: {len(negative_rag_ref_compilation)}')
        print(f'Error Type Count:\n{each_error_type_count}\n\n')
        print(f'Unknown error types: {unknow_types}')

        # print the negative rag ref compilation
        for each_tc_log_cov in negative_rag_ref_compilation:
            print(f'- target_focal_method:\n{each_tc_log_cov["target_coverage"]}')
            print(f'- target_test_case:\n{each_tc_log_cov["target_test_case"]}')

            print(f'- no_ref log_path:\n{each_tc_log_cov["log_path_no_ref"]}')
            print(f'- no_ref generation:\n{each_tc_log_cov["generation_no_ref"]}\n')

            print(f'- rag_ref log_path:\n{each_tc_log_cov["log_path_rag_ref"]}')
            print(f'- rag_ref generation:\n{each_tc_log_cov["generation_rag_ref"]}\n')

            for each_ref in each_tc_log_cov['rag_references']:
                print('-'*50)
                print(f'+ rag_ref score: {each_ref[0]}\n')
                print(f'+ rag_ref coverage:\n{each_ref[1]}\n')
                print(f'+ rag_ref test case:\n{each_ref[2]}\n')
            print('='*50)

    def get_positive_negative_rag_ref_coverage(self):
        is_common = True
        is_ref = 'no_ref'
        coverages = self.get_filtered_coverages(is_ref=is_ref, n_cover_line_threshold=1)

        if is_common:
            another_is_ref = 'rag_ref' if is_ref == 'no_ref' else 'no_ref'
            common_coverages = []
            for each_cov in coverages:
                if each_cov[f'coverage_{another_is_ref}'] is not None:
                    common_coverages.append(each_cov)
            coverages = common_coverages

        positive_cases, negative_cases = [], []

        for each_tc_log_cov in coverages:
            no_ref_cov = each_tc_log_cov['coverage_no_ref']
            rag_ref_cov = each_tc_log_cov['coverage_rag_ref']
            if no_ref_cov is None or rag_ref_cov is None:
                continue
            
            # target_focal_method_coverage, focal_file_coverage, focal_method_name

            is_exact_match_no_ref, is_fully_cover_no_ref, cover_ratio_no_ref, fm_cov_no_ref = self._analyze_cov(
                target_focal_method_coverage=each_tc_log_cov['target_coverage'], 
                focal_file_coverage=no_ref_cov, 
                focal_method_name=each_tc_log_cov['focal_method_name'],
                return_fm_cov=True
                )

            is_exact_match_rag_ref, is_fully_cover_rag_ref, cover_ratio_rag_ref, fm_cov_rag_ref = self._analyze_cov(
                target_focal_method_coverage=each_tc_log_cov['target_coverage'], 
                focal_file_coverage=rag_ref_cov, 
                focal_method_name=each_tc_log_cov['focal_method_name'],
                return_fm_cov=True
                )
            
            each_tc_log_cov['fm_cov_no_ref'] = fm_cov_no_ref
            each_tc_log_cov['fm_cov_rag_ref'] = fm_cov_rag_ref

            if is_exact_match_rag_ref > is_exact_match_no_ref or is_fully_cover_rag_ref > is_fully_cover_no_ref or cover_ratio_rag_ref > cover_ratio_no_ref:
                positive_cases.append(each_tc_log_cov)
            elif is_exact_match_rag_ref < is_exact_match_no_ref or is_fully_cover_rag_ref < is_fully_cover_no_ref or cover_ratio_rag_ref < cover_ratio_no_ref:
                negative_cases.append(each_tc_log_cov)
            
        print(f'Positive: {len(positive_cases)}, Negative: {len(negative_cases)}')
        for pos_neg_case in [('Positive', positive_cases), ('Negative', negative_cases)]:
            print(f'\n\n{pos_neg_case[0]} Cases:')  
            for each_case in pos_neg_case[1]:
                print(f'- target_focal_method:\n{each_case["target_coverage"]}')
                print(f'- target_test_case:\n{each_case["target_test_case"]}')

                print(f'- no_ref generation:\n{each_case["generation_no_ref"]}\n')
                print(f'- no_ref coverage:\n{each_case["fm_cov_no_ref"]}\n')

                print(f'- rag_ref generation:\n{each_case["generation_rag_ref"]}\n')
                print(f'- rag_ref coverage:\n{each_case["fm_cov_rag_ref"]}\n')

                for each_ref in each_case['rag_references']:
                    print('-'*50)
                    print(f'+ rag_ref score: {each_ref[0]}\n')
                    print(f'+ rag_ref coverage:\n{each_ref[1]}\n')
                    print(f'+ rag_ref test case:\n{each_ref[2]}\n')
                print('='*50)

    def cal_bleu(self, generated_target_pairs):
        generated_test_cases, target_test_cases = [], []
        for each_pair in generated_target_pairs:
            generated_test_cases.append(each_pair[0])
            target_test_cases.append(each_pair[1])

        bleu = evaluate.load('bleu')
        bleu_score = bleu.compute(predictions=generated_test_cases, references=target_test_cases)
        bleu_4 = bleu_score['precisions'][3]

        return bleu_4
    
    def get_filtered_coverages(self, is_ref, n_cover_line_threshold: int=1, is_common: bool=False):
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
        
        if is_common:
            another_is_ref = 'rag_ref' if is_ref == 'no_ref' else 'no_ref'
            common_coverages = []
            for each_cov in coverages:
                if each_cov[f'coverage_{another_is_ref}'] is not None:
                    common_coverages.append(each_cov)
            coverages = common_coverages

        return coverages
                
    def analyze_coverage(self, is_ref, n_cover_line_threshold: int=1, is_common: bool=False):
    # Args:
    #     is_ref: str, ['no_ref', 'human_ref', 'rag_ref']
    #     n_cover_line_threshold: int, the minimum number of covered lines in the target coverage.
    #     is_common: bool, whether to analyze the common coverage.
        assert is_ref in ['no_ref', 'rag_ref']
        assert n_cover_line_threshold > 0
        
        coverages = self.get_filtered_coverages(is_ref, n_cover_line_threshold, is_common=is_common)

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

            is_exact_match, is_fully_cover, cover_ratio = self._analyze_cov(target_focal_method_coverage, focal_file_coverage, focal_method_name)

            exact_match_cases.append(is_exact_match)
            fully_cover_cases.append(is_fully_cover)
            cover_ratio_list.append(cover_ratio)

        avg_exact_match = sum(exact_match_cases) / len(exact_match_cases)
        avg_cover = sum(fully_cover_cases) / len(fully_cover_cases)
        avg_cover_ratio = sum(cover_ratio_list) / len(cover_ratio_list)
        
        print(f'Exact Match: {avg_exact_match:.2%} ({sum(exact_match_cases)}/{len(exact_match_cases)})')
        print(f'Fully Cover: {avg_cover:.2%} ({sum(fully_cover_cases)}/{len(fully_cover_cases)})')
        print(f'Cover Ratio: {avg_cover_ratio:.2%}')
    
    def _analyze_cov(self, target_focal_method_coverage, focal_file_coverage, focal_method_name, return_fm_cov=False):        
        generated_focal_method_coverage = self.get_generated_focal_method_coverage(focal_file_coverage, target_focal_method_coverage, focal_method_name)

        is_exact_match, is_fully_cover, cover_ratio = self.eval_generated_coverage(generated_focal_method_coverage, target_focal_method_coverage)

        if return_fm_cov:
            return is_exact_match, is_fully_cover, cover_ratio, generated_focal_method_coverage
        else:
            return is_exact_match, is_fully_cover, cover_ratio

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
    
    def analyze_positive_reg_ref_apis(self, project_apis_extraction_save_path, project_dir):
        api_set = load_project_apis(project_apis_extraction_save_path, project_dir)

        positive_cases = self.get_positive_rag_ref_pass()

        test_case_no_ref_target_pairs, test_case_rag_ref_target_pairs = [], []
        for each_pos_case in positive_cases:
            target_test_case = each_pos_case['target_test_case']
            test_case_no_ref = each_pos_case['generation_no_ref']
            test_case_rag_ref = each_pos_case['generation_rag_ref']
            fm_name = each_pos_case['focal_method_name']

            test_case_no_ref_target_pairs.append([test_case_no_ref, target_test_case, fm_name])

            test_case_rag_ref_target_pairs.append([test_case_rag_ref, target_test_case, fm_name])
        
        print(f'\n\nAPI Analysis: Positive RAG Reference')
        self._analyze_apis(test_case_no_ref_target_pairs, test_case_rag_ref_target_pairs, api_set)

    def analyze_apis(self, project_apis_extraction_save_path, project_dir, is_pass, is_common):
        api_set = load_project_apis(project_apis_extraction_save_path, project_dir)

        test_case_no_ref_target_pairs, test_case_rag_ref_target_pairs = self.load_test_cases(is_pass=is_pass, is_common=is_common, return_focal_method=True)

        print(f'\n\nAPI Analysis: is_pass={is_pass}, is_common={is_common}')
        self._analyze_apis(test_case_no_ref_target_pairs, test_case_rag_ref_target_pairs, api_set)

    def _analyze_apis(self, test_case_no_ref_target_pairs, test_case_rag_ref_target_pairs, api_set):
        target_tc_api_count, no_ref_gen_tc_api_count, no_ref_gen_tc_api_cov_target_tc_api_ratio, target_tc_api_total, no_ref_gen_tc_api_total = self.extract_and_count_apis(test_case_no_ref_target_pairs, api_set)

        target_tc_api_count, rag_ref_gen_tc_api_count, rag_ref_gen_tc_api_cov_target_tc_api_ratio, target_tc_api_total, rag_ref_gen_tc_api_total = self.extract_and_count_apis(test_case_rag_ref_target_pairs, api_set)

        target_avg_gen_tc_api_count = sum(target_tc_api_count) / len(target_tc_api_count)

        no_ref_avg_gen_tc_api_count = sum(no_ref_gen_tc_api_count) / len(no_ref_gen_tc_api_count)
        no_ref_avg_api_cov_ratio = sum(no_ref_gen_tc_api_cov_target_tc_api_ratio) / len(no_ref_gen_tc_api_cov_target_tc_api_ratio)    
        
        rag_ref_avg_gen_tc_api_count = sum(rag_ref_gen_tc_api_count) / len(rag_ref_gen_tc_api_count)
        rag_ref_avg_api_cov_ratio = sum(rag_ref_gen_tc_api_cov_target_tc_api_ratio) / len(rag_ref_gen_tc_api_cov_target_tc_api_ratio)

        print(f'[Target] Avg #APIs/TC: {target_avg_gen_tc_api_count:.2f}')
        print(f'[Target] #APIs in Total: {target_tc_api_total}')
        
        print(f'[no_ref] Avg #APIs/TC: {no_ref_avg_gen_tc_api_count:.2f}')
        print(f'[no_ref] Avg API Coverage Ratio per TC: {no_ref_avg_api_cov_ratio:.2%}')
        print(f'[no_ref] #APIs in Total: {no_ref_gen_tc_api_total}')

        print(f'[rag_ref] Avg #APIs/TC: {rag_ref_avg_gen_tc_api_count:.2f}')
        print(f'[rag_ref] Avg API Coverage Ratio per TC: {rag_ref_avg_api_cov_ratio:.2%}')
        print(f'[rag_ref] #APIs in Total: {rag_ref_gen_tc_api_total}')
    
    def extract_and_count_apis(self, test_case_target_pairs, api_set):
        target_tc_api_count = []
        target_tc_in_project_api_set = set()
        gen_tc_api_count = []
        gen_tc_api_cov_target_tc_api_ratio = []  # the ratio of the number of APIs in the target test case that are also in the generated test case 
        gen_tc_in_project_api_set = set()

        for gen_tc, target_tc, fm_name in test_case_target_pairs:
            fm_name = fm_name.split('::::')[1].split('(')[0]
            target_tc_api = set(extract_method_invocation_from_java_code(target_tc))
            target_tc_api.discard(fm_name)

            gen_tc_api = set(extract_method_invocation_from_java_code(gen_tc))
            gen_tc_api.discard(fm_name)

            target_tc_in_project_api = target_tc_api & api_set
            target_tc_in_project_api_set.update(target_tc_in_project_api)

            gen_tc_in_project_api = gen_tc_api & api_set
            gen_tc_in_project_api_set.update(gen_tc_in_project_api)

            target_tc_api_count.append(len(target_tc_in_project_api))
            gen_tc_api_count.append(len(gen_tc_in_project_api))
            gen_tc_api_cov_target_tc_api_ratio.append(
                len(gen_tc_in_project_api & target_tc_in_project_api) / len(target_tc_in_project_api) if len(target_tc_in_project_api) > 0 else 0
                )
        return target_tc_api_count, gen_tc_api_count, gen_tc_api_cov_target_tc_api_ratio, len(target_tc_in_project_api_set), len(gen_tc_in_project_api_set)