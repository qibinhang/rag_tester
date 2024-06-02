import json


class Dataset:
    def __init__(self, configs):
        self.configs = configs
        self.raw_data = None
        self.coverage_data_human_labeled = None

    # TODO: the raw data format will be changed to # <Focal Method, Context, Test case>
    def load_raw_data(self):
        with open(self.configs.samples_path, 'r') as f:
            data = json.load(f)

        samples = []
        for focal_file_path, focal_methods in data.items():
            for each_focal_method in focal_methods:
                target_focal_method = each_focal_method['target_focal_method']
                target_test_case = each_focal_method['target_test_case']
                references = each_focal_method['references']
                samples.append((focal_file_path, target_focal_method, target_test_case, references))

        self.raw_data = samples
        return samples
    
    # TODO: check
    def load_coverage_data_human_labeled(self):
        # return: [<Coverage, Context, Test case>]
        with open(self.configs.coverage_data_human_labeled_path, 'r') as f:
            data = json.load(f)

        coverage_data = []
        for each_focal_file_path in data:
            each_focal_file_data = data[each_focal_file_path]
            for each_focal_method_name in each_focal_file_data:
                each_focal_method_data = each_focal_file_data[each_focal_method_name]
                for each_pair in each_focal_method_data:
                    each_test_case, each_coverage, each_context = each_pair
                    coverage_data.append(
                        (''.join(each_coverage).strip(), 
                         ''.join(each_context).strip(), 
                         ''.join(each_test_case).strip())
                         )
            
        self.coverage_data_human_labeled = coverage_data
        return coverage_data
    

if __name__ == '__main__':
    from configs import Configs
    configs = Configs('spark', 'charlie', 'llama_3', 'v0.9.2')
    dataset = Dataset(configs)
    # raw_data = dataset.load_raw_data()
    # print(raw_data)
    coverage_data = dataset.load_coverage_data_human_labeled()
    print(coverage_data)