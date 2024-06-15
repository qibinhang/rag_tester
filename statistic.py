import os
import re
import json
import evaluate


class Statistic():
    def __init__(self, configs):
        self.configs = configs

    def analyze_test_case_pass(self): 
        no_ref_log_names, human_ref_log_names, rag_ref_log_names = [], [], []
        for each_log in os.listdir(self.configs.test_case_run_log_dir):
            if 'no_ref' in each_log:
                no_ref_log_names.append(each_log)
            elif 'human_ref' in each_log:
                human_ref_log_names.append(each_log)
            else:
                rag_ref_log_names.append(each_log)
        # if len(no_ref_log_names) != len(human_ref_log_names):
        #         raise ValueError(f'len(no_ref_log_names) != len(human_ref_log_names). {len(no_ref_log_names)} != {len(human_ref_log_names)}')

        # if len(rag_ref_log_names) > 0:
        #     if len(no_ref_log_names) != len(rag_ref_log_names):
        #         raise ValueError(f'len(no_ref_log_names) != len(rag_ref_log_names). {len(no_ref_log_names)} != {len(rag_ref_log_names)}')
        print(f'No Reference: {len(no_ref_log_names)}')
        print(f'With Human Reference: {len(human_ref_log_names)}')
        print(f'With RAG Reference: {len(rag_ref_log_names)}')

        no_ref_fail_compile, no_ref_fail_execute, no_ref_success_pass, no_ref_fail_compile_count, no_ref_fail_execute_count, no_ref_success_pass_count = self.analyze_test_case_running_log(no_ref_log_names)

        if len(human_ref_log_names) > 0:
            human_ref_fail_compile, human_ref_fail_execute, human_ref_success_pass, human_ref_fail_compile_count, human_ref_fail_execute_count, human_ref_success_pass_count = self.analyze_test_case_running_log(human_ref_log_names)

        if len(rag_ref_log_names) > 0:
            rag_ref_fail_compile, rag_ref_fail_execute, rag_ref_success_pass, rag_ref_fail_compile_count, rag_ref_fail_execute_count, rag_ref_success_pass_count = self.analyze_test_case_running_log(rag_ref_log_names)

        print('\nSummary')
        print('-----------------------------------')
        print(f'[No Reference]\nFail Compile: {no_ref_fail_compile_count}, Fail Execute: {no_ref_fail_execute_count}, Success Pass: {no_ref_success_pass_count}({len(no_ref_success_pass)})\n')

        if len(human_ref_log_names) > 0:
            print(f'[With Human Reference]\nFail Compile: {human_ref_fail_compile_count}, Fail Execute: {human_ref_fail_execute_count}, Success Pass: {human_ref_success_pass_count}({len(human_ref_success_pass)})\n')

        if len(rag_ref_log_names) > 0:
            print(f'[With RAG Reference]\nFail Compile: {rag_ref_fail_compile_count}, Fail Execute: {rag_ref_fail_execute_count}, Success Pass: {rag_ref_success_pass_count}({len(rag_ref_success_pass)})')
        print('-----------------------------------\n')

        print('Detailed information:')
        print(f'[No Reference]\nFail Compile: {no_ref_fail_compile}\n\nFail Execute: {no_ref_fail_execute}\n\nSuccess Pass: {no_ref_success_pass}\n\n')
        
        if len(human_ref_log_names) > 0:
            print(f'[With Human Reference]\nFail Compile: {human_ref_fail_compile}\n\nFail Execute: {human_ref_fail_execute}\n\nSuccess Pass: {human_ref_success_pass}\n\n')

        if len(rag_ref_log_names) > 0:
            print(f'[With RAG Reference]\nFail Compile: {rag_ref_fail_compile}\n\nFail Execute: {rag_ref_fail_execute}\n\nSuccess Pass: {rag_ref_success_pass}\n\n')

    def analyze_test_case_running_log(self, log_names):
        fail_compile, fail_execute, success_pass = [], [], []
        fail_compile_count, fail_execute_count, success_pass_count = 0, 0, 0
        for each_log in log_names:
            running_log_path = os.path.join(self.configs.test_case_run_log_dir, each_log)
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
    
    def cal_bleu_for_saved_file(self):
        with open(self.configs.test_case_save_path, 'r') as f:
            generated_test_cases = json.load(f)

        test_case_no_ref_list, test_case_with_human_ref, test_case_with_rag_ref_list = [], [], []
        target_test_case_list = []
        for each_test_case in generated_test_cases:
            target_test_case = each_test_case['target_test_case']

            test_case_no_ref = each_test_case['generation_no_ref']
            test_case_no_ref_list.append(test_case_no_ref)

            test_case_with_rag_ref = each_test_case['generation_with_rag_ref']
            test_case_with_rag_ref_list.append(test_case_with_rag_ref)

            if each_test_case['generation_with_human_ref'] is not None:
                test_case_with_human_ref = each_test_case['generation_with_human_ref']
                test_case_with_human_ref.append(test_case_with_human_ref)

            target_test_case_list.append(target_test_case)
        
        bleu_no_ref = self.cal_bleu(test_case_no_ref_list, target_test_case_list)
        print(f'BLEU-4 without reference: {bleu_no_ref:.4f}')

        bleu_with_huamn_ref = None
        if len(test_case_with_human_ref) > 0:
            bleu_with_huamn_ref = self.cal_bleu(test_case_with_human_ref, target_test_case_list)
            print(f'BLEU-4 with human reference: {bleu_with_huamn_ref:.4f}')

        bleu_with_rag_ref = self.cal_bleu(test_case_with_rag_ref_list, target_test_case_list)
        print(f'BLEU-4 with RAG reference: {bleu_with_rag_ref:.4f}')

        return bleu_no_ref, bleu_with_huamn_ref, bleu_with_rag_ref

    def cal_bleu(self, generated_test_cases, target_test_cases):
        bleu = evaluate.load('bleu')
        bleu_score = bleu.compute(predictions=generated_test_cases, references=target_test_cases)
        bleu_4 = bleu_score['precisions'][3]

        return bleu_4
    
    def analyze_coverage(self, is_ref, n_cover_line_threshold: int=1):
        assert is_ref in ['no_ref', 'human_ref', 'rag_ref']
        assert n_cover_line_threshold > 0
        print(f'\n\nCoverage Analysis: = {is_ref} & Threshold {n_cover_line_threshold} =\n')  

        with open(self.configs.test_case_coverage_save_path, 'r') as f:
            total_coverage_infos = json.load(f)

        coverage_infos = dict()
        for test_case_running_log_name, cov_info in total_coverage_infos.items():
            if is_ref not in test_case_running_log_name:
                continue
            
            target_focal_method_coverage = cov_info['target_coverage']
            if target_focal_method_coverage.count('<COVER>') < n_cover_line_threshold:
                continue

            coverage_infos[test_case_running_log_name] = cov_info

        if len(coverage_infos) == 0:
            print(f'No target coverage has at least {n_cover_line_threshold} covered lines.')
            return

        exact_match_cases, fully_cover_cases, cover_ratio_list = [], [], []
        for test_case_running_log_name, cov_info in coverage_infos.items():
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
            

        
