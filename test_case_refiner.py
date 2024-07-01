import json
import os
from instruction_constructor import InstructionConstructor
from processing_tc import process_generated_test_case
from tqdm import tqdm


class TestCaseRefiner:
    def __init__(self, generator):
        self.is_ref_list = ['no_ref', 'rag_ref']
        self.instruction_constructor = InstructionConstructor()
        self.generator = generator
        
    def refine(self, test_case_log_and_coverage, is_ref):
        refined_test_cases = []

        for each_tc_log_cov in tqdm(test_case_log_and_coverage, ncols=100):
            if each_tc_log_cov[f'result_{is_ref}'] == 'SUCCESS':
                continue
            
            target_cov = each_tc_log_cov['target_coverage']
            target_context = each_tc_log_cov['target_context']
            focal_method_name = each_tc_log_cov['focal_method_name']
            fm_class_name = focal_method_name.split('::::')[0]

            generated_tc = each_tc_log_cov[f'generation_{is_ref}']
            generated_tc_error_msg = self.extract_error_message(each_tc_log_cov[f'log_path_{is_ref}'])

            init_refined_tc = self._refine(generated_tc, generated_tc_error_msg, target_cov, target_context, fm_class_name)

            refined_tc = process_generated_test_case(init_refined_tc, test_case_class_name=f'{fm_class_name}Test')

            if refined_tc is None:
                print(f'[WARNING] Abnormal refined test case: {init_refined_tc}') 
                continue

            test_case_dir = os.path.dirname(each_tc_log_cov[f'generation_{is_ref}_path'])
            test_case_path = f'{test_case_dir}/{fm_class_name}Test.java'

            each_tc_log_cov[f'before_refined_{is_ref}'] = {
                'error_msg': generated_tc_error_msg,
                'test_case': generated_tc,
                'test_case_path': each_tc_log_cov[f'generation_{is_ref}_path'],
                'result': each_tc_log_cov[f'result_{is_ref}']
            }

            each_tc_log_cov[f'generation_{is_ref}'] = refined_tc
            each_tc_log_cov[f'generation_{is_ref}_init'] = init_refined_tc
            each_tc_log_cov[f'generation_{is_ref}_path'] = test_case_path
            each_tc_log_cov[f'result_{is_ref}'] = None
            each_tc_log_cov[f'log_path_{is_ref}'] = None

            refined_test_cases.append(each_tc_log_cov)

        return refined_test_cases
    
    def save_refined_test_cases(self, refined_test_cases, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(refined_test_cases, f, indent=4)

    def _refine(self, generated_tc, generated_tc_error_msg, target_cov, target_context, focal_method_name):
        messages = self.instruction_constructor.instruct_for_refine_test_case(generated_tc, generated_tc_error_msg, target_cov, target_context, focal_method_name)
        
        refined_tc = self.generator._generate_test_case_using_llama3(messages)
        
        return refined_tc

    def extract_error_message(self, generated_test_case_log_path):
        with open(generated_test_case_log_path, 'r') as f:
            log = f.readlines()
        
        error_msg = []
        for each_line in log:
            if each_line.strip().startswith('[ERROR]'):
                if 'To see the full stack trace of the errors' in each_line:
                    break
                error_msg.append(each_line.replace('[ERROR]', '').strip())
        
        return '\n'.join(error_msg)
            