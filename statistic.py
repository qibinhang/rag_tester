import os
import re


class Statistic():
    def __init__(self, configs):
        self.configs = configs

    def analyze_test_case_pass(self): 
        no_ref_log_names, with_ref_log_names = [], []
        for each_log in os.listdir(self.configs.test_case_run_log_dir):
            if 'no_ref' in each_log:
                no_ref_log_names.append(each_log)
            else:
                with_ref_log_names.append(each_log)
        assert len(no_ref_log_names) == len(with_ref_log_names)

        no_ref_fail_compile, no_ref_fail_execute, no_ref_success_pass, no_ref_fail_compile_count, no_ref_fail_execute_count, no_ref_success_pass_count = self.analyze_test_case_running_log(no_ref_log_names)
        with_ref_fail_compile, with_ref_fail_execute, with_ref_success_pass, with_ref_fail_compile_count, with_ref_fail_execute_count, with_ref_success_pass_count = self.analyze_test_case_running_log(with_ref_log_names)

        print(f'[No Reference]\nFail Compile: {no_ref_fail_compile_count}, Fail Execute: {no_ref_fail_execute_count}, Success Pass: {no_ref_success_pass_count}\n\n')
        print(f'[With Reference]\nFail Compile: {with_ref_fail_compile_count}, Fail Execute: {with_ref_fail_execute_count}, Success Pass: {with_ref_success_pass_count}\n\n')

        print('Detailed information:')
        print(f'[No Reference]\nFail Compile: {no_ref_fail_compile}\n\nFail Execute: {no_ref_fail_execute}\n\nSuccess Pass: {no_ref_success_pass}\n\n')
        print(f'[With Reference]\nFail Compile: {with_ref_fail_compile}\n\nFail Execute: {with_ref_fail_execute}\n\nSuccess Pass: {with_ref_success_pass}\n\n')

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