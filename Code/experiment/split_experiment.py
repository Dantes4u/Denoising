import pandas as pd
import numpy as np
import os
import torch


class SplitExperiment:
    def __init__(self, domains, split_name):
        self.domains = domains
        self.split_name = split_name.capitalize()
        self.split_data = pd.DataFrame(dtype=np.int32)

    def update(self, label, loss):

        data = {}
        domain_name = self.domains[0]['name']
        if loss is not None:
            data['Loss'] = loss
        else:
            data['Loss'] = -1
        data[domain_name] = label.mean()
        self.split_data = pd.concat([self.split_data,pd.DataFrame([data])], sort=True, ignore_index=True)

    def get_split(self):
        split = {
            'name': self.split_name,
            'data': self.split_data,
        }
        return split

    def save(self, output_dir):
        split_data = pd.DataFrame(dtype=np.int32)
        for domain in self.domains:
            domain_name = domain['name']
            pred_name = domain_name + '-pred'
            for name in (domain_name, pred_name):
                split_data[name] = self.split_data[name].copy()
                split_data[name] = split_data[name].replace(dict(enumerate(domain['values'])))
        path = os.path.join(output_dir, self.split_name + '.csv')
        split_data.to_csv(path, index=False)



