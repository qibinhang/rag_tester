import os
from tqdm import tqdm


class TestCaseRunner():
    def __init__(self, configs):
        self.configs = configs

    def run_all_test_cases(self, test_cases):
        # run the generated test cases
        for each_test_case_path in tqdm(test_cases, ncols=80, desc='Running test cases'):
            test_case_no_ref_path, test_case_no_ref, test_case_with_ref_path, test_case_with_ref, focal_method_path = each_test_case_path

            print('Running the test case without reference...')
            os.makedirs(os.path.dirname(test_case_no_ref_path), exist_ok=True)
            with open(test_case_no_ref_path, 'w') as f:
                f.write(test_case_no_ref)

            self.run_test_case(test_case_no_ref_path, focal_method_path, is_ref=False)
            os.remove(test_case_no_ref_path)
            
            print('Running the test case with reference...')
            os.makedirs(os.path.dirname(test_case_with_ref_path), exist_ok=True)
            with open(test_case_with_ref_path, 'w') as f:
                f.write(test_case_with_ref)
            
            self.run_test_case(test_case_with_ref_path, focal_method_path, is_ref=True)
            os.remove(test_case_with_ref_path)


    def run_test_case(self, test_case_path, focal_method_path, is_ref):
        test_case_relative_path = test_case_path.replace(self.configs.project_test_case_base_path, '')
        test_case_relative_path = test_case_relative_path.replace('.java', '')
        test_case_relative_path = test_case_relative_path.replace('/', '.')

        focal_method_name = focal_method_path.split('/')[-1].split('.')[0]

        suffix = 'with_ref' if is_ref else 'no_ref'
        index = 1
        log_file_path = f'{self.configs.test_case_run_log_dir}/{focal_method_name}_{suffix}_{index}.log'
        while os.path.exists(log_file_path):
            index += 1
            log_file_path = f'{self.configs.test_case_run_log_dir}/{focal_method_name}_{suffix}_{index}.log'

        cmd = f'cd {self.configs.project_dir} && mvn clean verify -Dtest={test_case_relative_path} > {log_file_path} 2>&1'

        print(cmd)
        os.system(cmd)