import json
import re
import os
import sys
from tqdm import tqdm


def process_all_generated_test_cases(initial_test_case_save_path, processed_test_case_save_path):
    with open(initial_test_case_save_path, 'r') as f:
        test_cases = json.load(f)
    
    # save the generated test cases
    saved_test_cases = []  # will be saved to a json file
    for each_test_case in tqdm(test_cases, ncols=80, desc='Processing generated test cases'):
        focal_file_path = each_test_case['focal_file_path']
        focal_case_dir = focal_file_path[:focal_file_path.rfind('/')]
        # test_case_name = focal_file_path.split('/')[-1].split('.')[0]
        test_case_dir = focal_case_dir.replace('/main/', '/test/')
        fm_file_name = each_test_case['focal_method_name'].split('::::')[0]
        test_case_class_name = f'{fm_file_name}Test'

        test_case_no_ref = process_generated_test_case(each_test_case['generation_no_ref'], test_case_class_name)
        if test_case_no_ref is None:
            print(f'[WARNING] Abnormal test case: {focal_file_path}') 
            continue

        test_case_with_rag_ref = process_generated_test_case(each_test_case['generation_rag_ref'], test_case_class_name)
        if test_case_with_rag_ref is None:
            print(f'[WARNING] Abnormal test case: {focal_file_path}') 
            continue

        test_case_with_huam_ref = None
        if each_test_case['generation_human_ref'] is not None:
            test_case_with_huam_ref = process_generated_test_case(each_test_case['generation_human_ref'], test_case_class_name)
            if test_case_with_huam_ref is None:
                print(f'[WARNING] Abnormal test case: {focal_file_path}') 
                continue

        test_case_no_ref_path = f'{test_case_dir}/{test_case_class_name}.java'
        test_case_with_rag_ref_path = f'{test_case_dir}/{test_case_class_name}.java'
        test_case_with_huam_ref_path = f'{test_case_dir}/{test_case_class_name}.java' if test_case_with_huam_ref is not None else None

        each_test_case['focal_file_path'] = focal_file_path
        each_test_case['generation_no_ref_path'] = test_case_no_ref_path
        each_test_case['generation_no_ref'] = test_case_no_ref
        each_test_case['generation_human_ref_path'] = test_case_with_huam_ref_path
        each_test_case['generation_human_ref'] = test_case_with_huam_ref
        each_test_case['generation_rag_ref_path'] = test_case_with_rag_ref_path
        each_test_case['generation_rag_ref'] = test_case_with_rag_ref

        saved_test_cases.append(each_test_case)

    os.makedirs(os.path.dirname(processed_test_case_save_path), exist_ok=True)
    with open(processed_test_case_save_path, 'w') as f:
        json.dump(saved_test_cases, f, indent=4)


def process_generated_test_case(init_generation, test_case_class_name):
    test_case = _extract_test_case(init_generation, test_case_class_name)
    if test_case is None:
        return None
    
    test_case = _remove_assertions(test_case)
    # class_name = _get_class_name(test_case)
    
    # return test_case, class_name
    return test_case


def _extract_test_case(init_generation, test_case_class_name):
    # extract the test case from the generation
    result = re.findall(r'```java\n(.*?)```', init_generation, re.DOTALL)
    if len(result) == 0:
        result = re.findall(r'```\n(.*?)```', init_generation, re.DOTALL)
    if len(result) == 0:
        print('[WARNING] Abnormal generated test case:\n', init_generation, '\n\n')
        return None

    test_case = None
    for each_code in result:
        if '@Test' in each_code and test_case_class_name in each_code:
            test_case = each_code
            break
    if test_case is None:
        print('[WARNING] does not meet the requirement (@Test in each_code and test_case_class_name in each_code):\n', init_generation, '\n\n')
        return None
    
    return test_case


def _get_class_name(test_case):
    # get the class name of the test case
    class_name = re.search(r'\sclass\s+(.+?)\s', test_case)
    if class_name is not None:
        class_name = class_name.group(1)
        if 'Test' not in class_name:
            raise ValueError(f'Invalid class name in the generated test case:\n{test_case}. Maybe need manually check the extraction.')
    else:
        print('[WARNING] Cannot find the class name in the generated test case:\n', test_case, '\n\n')
        # find the method name
        class_name = re.search(r'\s(.+?)\s*\(', test_case)
        if class_name is not None:
            class_name = class_name.group(1)
        else:
            raise('[WARNING] Cannot find the method name in the generated test case:\n', test_case, '\n\n')
    return class_name


def _remove_assertions(test_case):
    assertion_patterns = [
        r'\bAssert\.assert.*?\(.*?\);',
        r'\bassert.*?\(.*?\);',
    ]
    processed_test_case_lines = []
    test_case_lines = test_case.split('\n')
    for each_line in test_case_lines:
        if re.search(assertion_patterns[0], each_line) or re.search(assertion_patterns[1], each_line):
            print(f'Removed assertion line: {each_line}')
        else:
            processed_test_case_lines.append(each_line)
    
    processed_test_case = '\n'.join(processed_test_case_lines)
    if processed_test_case != test_case:
        print(f'Original test case:\n{test_case}\n\n')
        print(f'Processed test case:\n{processed_test_case}\n\n')

    return processed_test_case