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
        if len(no_ref_log_names) != len(human_ref_log_names):
                raise ValueError(f'len(no_ref_log_names) != len(human_ref_log_names). {len(no_ref_log_names)} != {len(human_ref_log_names)}')

        if len(rag_ref_log_names) > 0:
            if len(no_ref_log_names) != len(rag_ref_log_names):
                raise ValueError(f'len(no_ref_log_names) != len(rag_ref_log_names). {len(no_ref_log_names)} != {len(rag_ref_log_names)}')

        no_ref_fail_compile, no_ref_fail_execute, no_ref_success_pass, no_ref_fail_compile_count, no_ref_fail_execute_count, no_ref_success_pass_count = self.analyze_test_case_running_log(no_ref_log_names)

        human_ref_fail_compile, human_ref_fail_execute, human_ref_success_pass, human_ref_fail_compile_count, human_ref_fail_execute_count, human_ref_success_pass_count = self.analyze_test_case_running_log(human_ref_log_names)

        if len(rag_ref_log_names) > 0:
            rag_ref_fail_compile, rag_ref_fail_execute, rag_ref_success_pass, rag_ref_fail_compile_count, rag_ref_fail_execute_count, rag_ref_success_pass_count = self.analyze_test_case_running_log(rag_ref_log_names)

        print('\nSummary')
        print('-----------------------------------')
        print(f'[No Reference]\nFail Compile: {no_ref_fail_compile_count}, Fail Execute: {no_ref_fail_execute_count}, Success Pass: {no_ref_success_pass_count}({len(no_ref_success_pass)})\n')
        print(f'[With Human Reference]\nFail Compile: {human_ref_fail_compile_count}, Fail Execute: {human_ref_fail_execute_count}, Success Pass: {human_ref_success_pass_count}({len(human_ref_success_pass)})\n')

        if len(rag_ref_log_names) > 0:
            print(f'[With RAG Reference]\nFail Compile: {rag_ref_fail_compile_count}, Fail Execute: {rag_ref_fail_execute_count}, Success Pass: {rag_ref_success_pass_count}({len(rag_ref_success_pass)})')
        print('-----------------------------------\n')

        print('Detailed information:')
        print(f'[No Reference]\nFail Compile: {no_ref_fail_compile}\n\nFail Execute: {no_ref_fail_execute}\n\nSuccess Pass: {no_ref_success_pass}\n\n')
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

        test_case_no_ref_list, test_case_with_ref_list, test_case_with_rag_ref_list = [], [], []
        target_test_case_list = []
        for each_test_case in generated_test_cases:
            focal_method_path, test_case_no_ref_path, test_case_no_ref, test_case_with_ref_path, test_case_with_ref, test_case_with_rag_ref_path, test_case_with_rag_ref, target_test_case = each_test_case
            
            test_case_no_ref_list.append(test_case_no_ref)
            test_case_with_ref_list.append(test_case_with_ref)
            test_case_with_rag_ref_list.append(test_case_with_rag_ref)

            target_test_case_list.append(target_test_case)
        
        bleu_no_ref = self.cal_bleu(test_case_no_ref_list, target_test_case_list)
        bleu_with_ref = self.cal_bleu(test_case_with_ref_list, target_test_case_list)
        bleu_with_rag_ref = self.cal_bleu(test_case_with_rag_ref_list, target_test_case_list)

        print(f'BLEU-4 without reference: {bleu_no_ref:.4f}')
        print(f'BLEU-4 with human reference: {bleu_with_ref:.4f}')
        print(f'BLEU-4 with RAG reference: {bleu_with_rag_ref:.4f}')

        return bleu_no_ref, bleu_with_ref, bleu_with_rag_ref

    def cal_bleu(self, generated_test_cases, target_test_cases):
        bleu = evaluate.load('bleu')
        bleu_score = bleu.compute(predictions=generated_test_cases, references=target_test_cases)
        bleu_4 = bleu_score['precisions'][3]

        return bleu_4