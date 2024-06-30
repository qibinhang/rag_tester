import json
import os
from collections import namedtuple
CoveragePair = namedtuple('CoveragePair', ['project_name', 'focal_file_path', 'focal_method_name', 'coverage', 'focal_method', 'context', 'test_case', 'references'])


class Dataset:
    def __init__(self, configs):
        self.configs = configs
        self.raw_data = None
        self.coverage_human_labeled = None

    def load_focal_method_data(self):
        path = os.path.join(self.configs.collected_focal_method_dir, f'{self.configs.project_name}.json')
        coverage_data = self._load_coverage_data_jacoco(path)
        return coverage_data

    # TODO: add the references: [[tc1, tc2, ...], [cov1, cov2, ...]]
    def load_coverage_data_jacoco(self):
        path = os.path.join(self.configs.coverage_human_labeled_dir, f'{self.configs.project_name}.json')
        coverage_data = self._load_coverage_data_jacoco(path)
        return coverage_data

    def _load_coverage_data_jacoco(self, path: str):
        coverage_data = []
        with open(path, 'r') as f:
            data = json.load(f)
        for each_focal_file_path, coverages in data.items():
            for each_fm_name, tc_cov_pairs in coverages.items():
                for each_pair in tc_cov_pairs:
                    tc, cov, context = each_pair
                    fm = ''.join(cov).replace('<COVER>', '')

                    coverage_pair = CoveragePair(
                        project_name=self.configs.project_name, 
                        focal_file_path=each_focal_file_path, 
                        focal_method=fm, 
                        coverage=''.join(cov), 
                        context=''.join(context), 
                        test_case=''.join(tc), 
                        focal_method_name=each_fm_name, 
                        references=None
                    )
                    coverage_data.append(coverage_pair)
        return coverage_data

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
    
    def load_coverage_data(self, label_method: str):
        assert label_method in ['human', 'model']
        if label_method == 'human':
            coverage_data = self._load_coverage_data(self.configs.coverage_human_labeled_dir)
            self.coverage_human_labeled = coverage_data
            return coverage_data
        else:
            raise NotImplementedError('Model labeled coverage data is not implemented yet.')

    def load_unlabeled_coverage_data(self):
        unlabeled_coverage_data = self._load_coverage_data(self.configs.coverage_data_model_unlabeled_path)
        return unlabeled_coverage_data

    def _load_coverage_data(self, coverage_dir: str):
        # return: [<Coverage, Context, Test case>] or [<Focal method, Context, Test case>]
        # list the files in the directory
        coverage_data = []

        file_names = os.listdir(coverage_dir)
        for each_file_name in file_names:
            path = os.path.join(coverage_dir, each_file_name)

            with open(path, 'r') as f:
                data = json.load(f)

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
            
        return coverage_data
    

if __name__ == '__main__':
    from configs import Configs
    configs = Configs('spark', 'charlie', 'llama_3', 'v0.9.2')
    dataset = Dataset(configs)
    # raw_data = dataset.load_raw_data()
    # print(raw_data)
    coverage_data = dataset.load_coverage_data(label_method='human')
    print(coverage_data)