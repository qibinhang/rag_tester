import os


class Configs:
    def __init__(self, project_name, environment, llm_name, version) -> None:
        assert environment in ['charlie', 'cluster']
        self.project_name = project_name
        self.environment = environment
        self.llm_name = llm_name
        self.version = version

        self.root_dir = '/evosuite_pp' if environment == 'charlie' else '/home/q/qibh/Documents'

        # format: [{focal_file_path: list([target_focal_method, target_test_case, references])
        self.samples_path = f'{self.root_dir}/rag_tester/data/samples_with_reference/medium_cases_{project_name}_reformat.json'
        self.project_dir = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/{project_name}'

        if project_name == 'spark':
            self.project_test_case_base_path = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/spark/src/test/java/spark'
        elif project_name == 'HdrHistogram':
            self.project_test_case_base_path = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/HdrHistogram/src/test/java/org/HdrHistogram'
        else:
            raise ValueError('Invalid project name')
        
        # format: list([focal_file_path, generation_no_ref, generation_with_human_ref, generation_with_rag_ref, target_test_case)]
        self.test_case_initial_gen_save_path = f'./data/generated_test_cases/{project_name}_{llm_name}_init_gen_{version}.json'

        # the format of saved test cases:
        # list([focal_method_path, test_case_no_ref_path, test_case_no_ref, test_case_with_ref_path, test_case_with_ref, test_case_with_rag_ref_path, test_case_with_rag_ref])
        self.test_case_save_path = f'./data/generated_test_cases/{project_name}_{llm_name}_processed_{version}.json'

        self.test_case_run_log_dir = os.path.abspath(f'./data/generated_test_cases_run_log/{project_name}_{llm_name}_{version}')

        self.coverage_data_human_labeled_path = f'./data/{project_name}/coverage_human_labeled.json'