import os
import json
import shutil
from tqdm import tqdm
from bs4 import BeautifulSoup


class TestCaseRunner():
    def __init__(self, configs, test_case_run_log_dir):
        self.configs = configs
        self.test_case_run_log_dir = test_case_run_log_dir
        self.cur_no_ref_log_name = None
        self.cur_human_ref_log_name = None
        self.cur_rag_ref_log_name = None

        self.focal_file_coverage = dict()  # e.g., {'Base64_1_no_ref': cov_no_ref, 'Base64_1_with_rag_ref': cov_with_rag_ref}

    def run_all_test_cases(self, test_cases, is_ref):
        test_case_with_log_coverage = []
        # run the generated test cases
        for each_test_case in tqdm(test_cases, ncols=80, desc='Running test cases'):
            focal_file_path = each_test_case['focal_file_path']

            generation_relative_path = each_test_case[f'generation_{is_ref}_path']
            tc_path = f"{self.configs.project_dir}/{generation_relative_path}"
            tc = each_test_case[f'generation_{is_ref}']

            log_path, focal_file_coverage = self.run_test_case_and_get_coverage(tc, tc_path, focal_file_path, is_ref=is_ref)
            each_test_case[f'log_path_{is_ref}'] = log_path
            each_test_case[f'coverage_{is_ref}'] = focal_file_coverage

            test_case_with_log_coverage.append(each_test_case)
        return test_case_with_log_coverage

    def save_log_coverage(self, log_coverage, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(log_coverage, f, indent=4)
            print(f"Saved the generated test cases' log and coverage to {save_path}")

    def run_test_case(self, test_case_path, focal_file_path, is_ref):
        assert is_ref in ('no_ref', 'human_ref', 'rag_ref')
        test_case_relative_path = self.get_test_case_relative_path(test_case_path)

        focal_method_name = focal_file_path.split('/')[-1].split('.')[0]

        suffix = is_ref
        index = 1
        log_file_path = f'{self.test_case_run_log_dir}/{focal_method_name}_{index}_{suffix}.log'
        while os.path.exists(log_file_path):
            index += 1
            log_file_path = f'{self.test_case_run_log_dir}/{focal_method_name}_{index}_{suffix}.log'
        setattr(self, f'cur_{is_ref}_ref_log_name', f'{focal_method_name}_{index}_{suffix}')

        cmd = f'cd {self.configs.project_dir}/{self.configs.project_name} && mvn clean verify -Dtest={test_case_relative_path} > {log_file_path} 2>&1'

        print(cmd)
        os.system(cmd)
        return log_file_path

    def run_test_case_and_get_coverage(self, test_case, test_case_path, focal_file_path, is_ref):
        print(f'Running the test case with = {is_ref} = reference...')
        # remove the folder of test cases
        tc_rel_path = test_case_path.split('/src/test/')[1]
        tc_base_dir = test_case_path.replace(tc_rel_path, '')
        if os.path.exists(tc_base_dir):
            shutil.rmtree(tc_base_dir)
        os.makedirs(os.path.dirname(test_case_path), exist_ok=True)
        with open(test_case_path, 'w') as f:
            f.write(test_case)

        tc_run_log_path = self.run_test_case(test_case_path, focal_file_path, is_ref)

        focal_file_coverage = self.get_focal_file_coverage(focal_file_path, test_case_path)
        focal_file_coverage = ''.join(focal_file_coverage) if focal_file_coverage is not None else None

        return tc_run_log_path, focal_file_coverage

    def get_test_case_relative_path(self, test_case_path):
        test_case_relative_path = test_case_path.split('/src/test/java/')[1]
        test_case_relative_path = test_case_relative_path.split('/')[1:]
        test_case_relative_path = '/'.join(test_case_relative_path)
        test_case_relative_path = test_case_relative_path.replace('.java', '')
        test_case_relative_path = test_case_relative_path.replace('/', '.')
        return test_case_relative_path

    def get_focal_file_coverage(self, focal_file_path, test_case_path):
        base_path = f'{self.configs.project_dir}/{self.configs.project_name}'
        org_name = test_case_path.split('/src/test/java/')[1].split('/')[0]
        test_suffix = 'Test'
        test_case_relative_path = self.get_test_case_relative_path(test_case_path)

        jacoco_report_path = self.get_jacoco_report_path(base_path, test_case_relative_path, org_name, test_suffix)

        if not os.path.exists(jacoco_report_path):
            print(f'[WARNING] Jacoco report not found: {jacoco_report_path}')
            return None

        cov_lines, uncov_lines = self.get_lines_coverage(jacoco_report_path)
        with open(f'{self.configs.project_dir}/{focal_file_path}', 'r') as f:
            focal_file = f.readlines()
        for line in cov_lines:
            if focal_file[line - 1].strip() != '}':
                focal_file[line - 1] = "<COVER>" + focal_file[line - 1]
        return focal_file

    # copy from /bernard/dataset_construction/human_written_tests/v2/utils.py
    def get_jacoco_report_path(self, base_path, test_class_name, org_name, test_suffix):
        # get jacoco report
        # append_path = "spark/" if '.' not in test_class_name else "spark." + '.'.join(test_class_name.split(".")[:-1]) + '/'
        append_path = org_name + "/" if '.' not in test_class_name else org_name + "." + '.'.join(test_class_name.split(".")[:-1]) + '/'
        suff_len = len(test_suffix)
        html_name = test_class_name.split(".")[-1][:suff_len * -1] + ".java.html" # changes from -4 to -5 depending on whether it's Test or Tests
        
        jacoco_path = base_path + "/target/site/jacoco/" + append_path + html_name
        return jacoco_path

    # copy from /bernard/dataset_construction/human_written_tests/v2/utils.py
    def get_lines_coverage(self, jacoco_report_path):
        with open(jacoco_report_path) as f:
            soup = BeautifulSoup(f, 'html.parser')
            # find all spans with class 'fc' or 'pc' or 'bpc', and extract the ID
            cov_lines = []
            uncov_lines = []
            for span in soup.find_all('span', class_=['fc', 'pc', 'bpc', 'nc']):
                if span['class'][0] == 'nc':
                    uncov_lines.append(int(span['id'][1:]))
                else:
                    cov_lines.append(int(span['id'][1:]))
        
        return cov_lines, uncov_lines