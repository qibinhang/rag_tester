import os


class Configs:
    def __init__(self) -> None:
        self.environment = ['charlie', 'cluster'][0]
        self.project_name = ['spark', 'HdrHistogram', 'blade/blade-kit', 'hutool/hutool-core'][2]
        self.llm_name = ['llama_3', 'llama_3:70b'][0]
        self.retrieval_mode = ['fm', 'tc', 'both'][0]
        self.retrieval_top_k = 1

        self.version = f'v0.11.0_mode_{self.retrieval_mode}_k{self.retrieval_top_k}'
        self.version_intro = 'analyze the coverage of generated test case and compare it with the desired coverage.'
        
        self.max_context_len = 1024
        self.max_input_len = 3072
        self.max_num_generated_tokens = 1024
        self.do_sample = False
        self.top_p = None
        self.tempurature = None
        self.verbose = True

        self.root_dir = '/evosuite_pp' if self.environment == 'charlie' else '/home/q/qibh/Documents'

        # format: [{focal_file_path: list([target_focal_method, target_test_case, references])
        self.samples_path = f'{self.root_dir}/rag_tester/data/samples_with_reference/medium_cases_{self.project_name}_reformat.json'
        self.project_dir = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test'

        if self.project_name == 'spark':
            self.project_test_case_base_path = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/spark/src/test/java/spark'
        elif self.project_name == 'HdrHistogram':
            self.project_test_case_base_path = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/HdrHistogram/src/test/java/org'
        elif self.project_name == 'blade/blade-kit':
            self.project_test_case_base_path = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/blade/blade-kit/src/test/java/com'
        elif self.project_name == 'hutool/hutool-core':
            self.project_test_case_base_path = f'{self.root_dir}/rag_tester/data/raw_data/repos_removing_test/hutool/hutool-core/src/test/java/cn'
        else:
            raise ValueError('Invalid project name')
        
        # format: list([focal_file_path, generation_no_ref, generation_with_human_ref, generation_with_rag_ref, target_test_case)]
        self.test_case_initial_gen_save_path = f'{self.root_dir}/rag_tester/data/generated_test_cases/{self.project_name}_{self.llm_name}_init_gen_{self.version}.json'

        # the format of saved test cases:
        # list([focal_method_path, test_case_no_ref_path, test_case_no_ref, test_case_with_ref_path, test_case_with_ref, test_case_with_rag_ref_path, test_case_with_rag_ref])
        self.test_case_save_path = f'{self.root_dir}/rag_tester/data/generated_test_cases/{self.project_name}_{self.llm_name}_processed_{self.version}.json'

        self.test_case_run_log_dir = os.path.abspath(f'{self.root_dir}/rag_tester/data/generated_test_cases_run_log/{self.project_name}_{self.llm_name}_{self.version}')

        self.test_case_coverage_save_path = f'{self.root_dir}/rag_tester/data/generated_test_cases_coverage/{self.project_name}_{self.llm_name}_{self.version}.json'

        self.coverage_human_labeled_dir = f'{self.root_dir}/rag_tester/data/coverage_human_labeled'
        self.coverage_model_unlabeled_dir = f'{self.root_dir}/rag_tester/data/coverage_model_unlabeled'
        self.coverage_model_labeled_dir = f'{self.root_dir}/rag_tester/data/coverage_model_labeled'

        self.set_retriever_configs()
        self.set_coverage_labeling_model_configs()
        self.set_coverage_labeling_predictor_configs()

    def set_coverage_labeling_model_configs(self):
        # the major configures are in instruction_tuning_configs.yaml
        self.labeling_model_path = f'{self.root_dir}/rag_tester/data/llama_3_8b_coverage_prediction'

    def set_coverage_labeling_predictor_configs(self):
        self.predictor_model_dir = f'{self.root_dir}/rag_tester/data/predictor'

    def set_retriever_configs(self):
        self.retriever_model_name = 'Salesforce/codet5p-770m'
        self.retriever_model_dir = f'{self.root_dir}/rag_tester/data/retriever'
        
        self.retriever_database_fm2cov_emb_path = f'{self.retriever_model_dir}/database_fm2cov_embeddings.pkl'
        self.retriever_database_fm2cov_bm25_path = f'{self.retriever_model_dir}/database_fm2cov_bm25.pkl'

        self.retriever_database_cov2tc_emb_path = f'{self.retriever_model_dir}/database_cov2tc_embeddings.pkl'
        self.retriever_database_cov2tc_bm25_path = f'{self.retriever_model_dir}/database_cov2tc_bm25.pkl'
        # self.retriever_model_path = f'{self.retriever_model_dir}/'
        
        self.retriever_n_epochs = 10
        self.retriever_lr = 5e-5
        self.retriever_train_batch_size = 8

        self.retriever_max_source_length = 768
        self.retriever_max_target_length = 600  # according to codet5p paper

        self.gradient_accumulation_steps = 4
