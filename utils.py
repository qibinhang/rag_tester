import os
import json
import matplotlib.pyplot as plt
import javalang
from dataset import Dataset
from configs import Configs
from tqdm import tqdm


def load_project_apis(project_apis_extraction_save_path, project_dir):
    if os.path.exists(project_apis_extraction_save_path):
        with open(project_apis_extraction_save_path, 'r') as f:
            project_apis = json.load(f)
    else:
        project_apis = extract_method_declaration_from_java_project(project_dir)
        os.makedirs(os.path.dirname(project_apis_extraction_save_path), exist_ok=True)
        with open(project_apis_extraction_save_path, 'w') as f:
            json.dump(project_apis, f)

    api_set = set()
    for each_file in project_apis:
        apis = set(each_file['apis'])
        api_set.update(apis)

    print(f'In-project API declarations: {len(api_set)}')
    return api_set


def extract_method_declaration_from_java_project(project_path):
    api_collection = []
    file_path_list = []
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if file.endswith('.java'):
                file_path = os.path.join(root, file)
                file_path_list.append(file_path)

    n_exceptions = 0
    for file_path in tqdm(file_path_list, desc='Extracting APIs', ncols=100):
        api_info = extract_method_declaration_from_java_file(file_path)
        if len(api_info) == 0:
            n_exceptions += 1
            continue
        api_collection.append({'file_path': file_path, 'apis': api_info})

    print(f'Number of files cannot extract API declarations: {n_exceptions}')
    return api_collection
    

def extract_method_declaration_from_java_file(file_path):
    with open(file_path, 'r') as file:
        java_code = file.read()
    return extract_method_declaration_from_java_code(java_code)


def extract_method_declaration_from_java_code(java_code):
    api_info = []

    try:
        tree = javalang.parse.parse(java_code)
    except Exception as e:
        return api_info
    
    for path, node in tree:
        if isinstance(node, javalang.tree.MethodDeclaration):
            method_name = node.name

            # modifiers = ' '.join(node.modifiers)
            # return_type = node.return_type.name if node.return_type else 'void'
            # parameters = ', '.join([f"{param.type.name} {param.name}" for param in node.parameters])
            # method_signature = f"{modifiers} {return_type} {method_name}({parameters})"
            # api_info.append(method_signature)

            api_info.append(method_name)
    
    return api_info


def extract_method_invocation_from_java_code(java_code):
    api_info = []

    try:
        tree = javalang.parse.parse(java_code)
    except Exception as e:
        return api_info
    
    for path, node in tree:
        if isinstance(node, javalang.tree.MethodInvocation):
            method_name = node.member

            # qualifier = f"{node.qualifier}." if node.qualifier else ""
            # invocation = f"{qualifier}{node.member}()"
            # api_info.append(invocation)

            api_info.append(method_name)
    
    return api_info


def load_coverage_pairs(file_path, filename):
    configs.project_name = filename.split('.')[0]
    dataset = Dataset(configs)
    coverages = dataset._load_coverage_data_jacoco(file_path)
    return coverages


def count_coverage_pairs(coverage_dir):
    total_num = 0
    count = dict()
    for root, dirs, files in os.walk(coverage_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            coverages = load_coverage_pairs(file_path, filename)
            count[filename.split('.')[0]] = len(coverages)
            total_num += len(coverages)
    
    print('Total number of coverage pairs:', total_num)
    for key, value in count.items():
        print(f'{key}: {value}')
    
    # more than 20
    print('\nMore than 20:')
    for key, value in count.items():
        if value > 20:
            print(f'{key}: {value}')

    keys = list(count.keys())
    values = list(count.values())
    plt.bar(keys, values)
    plt.show()


def get_negative_rag_ref(self):
    print('\n\nNegative RAG Reference: ')

    for each_tc_log_cov in self.test_case_log_analysis:
        if each_tc_log_cov['result_no_ref'] == 'SUCCESS' and each_tc_log_cov['result_rag_ref'] != 'SUCCESS':
            print(f'- no_ref:\n{each_tc_log_cov["generation_no_ref"]}')
            print(f'- rag_ref:\n{each_tc_log_cov["generation_rag_ref"]}\n')
            print(f'- rag_references:\n{each_tc_log_cov["target_test_case"]}\n\n')


if __name__ == '__main__':
    configs = Configs('spark')
    # coverage_dir = '/evosuite_pp/rag_tester/data/coverage_human_labeled'
    # count_coverage_pairs(coverage_dir)
