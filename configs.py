class Configs:
    def __init__(self, project_name, environment) -> None:
        assert environment in ['charlie', 'cluster']

        self.root_dir = '/evosuite_pp' if environment == 'charlie' else '/home/q/qibh/Documents'

        self.samples_path = f'{self.root_dir}/rag_tester/data/samples_with_reference/medium_cases_{project_name}_reformat.json'
        self.project_dir = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/{project_name}'

        if project_name == 'spark':
            self.project_test_case_base_path = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/spark/src/test/java/spark'
        elif project_name == 'HdrHistogram':
            self.project_test_case_base_path = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/HdrHistogram/src/test/java/org/HdrHistogram'
        else:
            raise ValueError('Invalid project name')
        