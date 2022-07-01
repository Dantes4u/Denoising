import os
import json
import numpy as np
import torch


class DataHolder:
    def __init__(self, config):
        metadata_dir = config['metadata_dir']
        train_datasets = config['train_datasets']
        test_datasets = config['test_datasets']

        self.data_dir = config['data_dir']
        self.datasets = {}
        self.train_labels = []
        self.domains = config['domains']

        for input_datasets in (train_datasets, test_datasets):
            for dataset in input_datasets:
                self.datasets[dataset] = {}
                with open(os.path.join(metadata_dir, dataset + '.json'), 'r') as input_file:
                    input_data = json.load(input_file)
                count = 0
                for filepath, data in input_data.items():
                    if data['use']:
                        self.datasets[dataset][count] = {}
                        self.datasets[dataset][count]['clean'] = f"clean/{data['path']}"
                        self.datasets[dataset][count]['noisy'] = f"noisy/{data['path']}"
                        count+=1

    def get_dataset(self, dataset_name):
        dataset = {
            'name': dataset_name,
            'data': self.datasets[dataset_name],
            'size': len(self.datasets[dataset_name]),
            'train': 'train' in dataset_name
        }

        return dataset


