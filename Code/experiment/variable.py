import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from scipy import interp

from experiment import update_var_metrics


class VariablePlot:
    def __init__(self, name, settings, domain=None):
        self.settings = settings
        self.basic_render = settings.get('basic_render', False)
        self.name = name
        self.domain = domain
        self.variable = {'split_name': [], 'epoch': [], self.name: []}
        self.epochs = set()

    def set(self, split_name, data, epoch):
        if self.name in ['Loss']:
            if self.domain is None:
                field_name = self.name
                data = data.loc[:, [self.name]]
            else:
                field_name = self.domain['name'] + '-dice'
            data = data.loc[:, [field_name]]
            data = data[data[field_name] != self.settings['ignore_value']]
            if not data.empty:
                self._set_values(split_name, epoch, data.mean().values[0])
        else:
            domain_name = self.domain['name']
            pred_name = domain_name + '-pred'
            data = data[data[domain_name] != self.settings['ignore_value']].copy()
            if not data.empty:
                if self.name.endswith('MAE'):
                    data.loc[:, 'Value'] = np.abs(data[domain_name] - data[pred_name])
                elif self.name.endswith('Accuracy'):
                    data.loc[:, 'Value'] = (data[domain_name] == data[pred_name])
                elif self.name == 'ROC':
                    fpr = dict()
                    tpr = dict()
                    num_values = len(self.domain['values'])
                    if num_values != 2:
                        bin_data = label_binarize(data[domain_name].values, classes=range(num_values))
                    else:
                        bin_data = np.array([[1, 0] if value == 0 else [0, 1] for value in data[domain_name].values])
                    for i, value in enumerate(self.domain['values']):
                        scores = data[domain_name + '-scores-{}'.format(i)].values
                        fpr[value], tpr[value], _ = roc_curve(bin_data[:, i], scores)
                    all_fpr = np.unique(np.concatenate([fpr[value] for value in self.domain['values']]))
                    mean_tpr = np.zeros_like(all_fpr)
                    mean_tpr_count = 0
                    for value in self.domain['values']:
                        if not np.isnan(tpr[value]).all():
                            mean_tpr += interp(all_fpr, fpr[value], tpr[value])
                            mean_tpr_count += 1
                    mean_tpr /= mean_tpr_count
                    data.loc[:, 'Value'] = auc(all_fpr, mean_tpr)
                elif self.name == 'PR' and self.domain['length'] == 2:
                    precision, recall, thresholds = precision_recall_curve(
                        data[domain_name].values,
                        data[domain_name + '-scores-1'].values
                    )
                    for i in range(len(precision)):
                        precision[i] = np.max(precision[:i+1])
                    data.loc[:, 'Value'] = -np.sum(np.diff(recall) * np.array(precision)[:-1])

                if 'Value' in data:
                    if self.name.startswith('M-'):
                        data = data.loc[:, [domain_name, 'Value']]
                        data = data.groupby([domain_name]).mean()
                    value = np.mean(data['Value'])
                else:
                    value = None

                self._set_values(split_name, epoch, value)

    def _set_values(self, split_name, epoch, value):
        if value is not None:
            self.variable['split_name'].append(split_name)
            self.variable['epoch'].append(epoch)
            self.variable[self.name].append(value)
            self.epochs.add(epoch)

    def render(self, output_dirs, metrics):
        if not self.variable[self.name]:
            return
        data = pd.DataFrame(self.variable).astype({self.name: 'float32'})

        if self.basic_render:
            data = data.loc[data['split_name'].isin(('Overall', 'Train'))]

        fig, ax = plt.subplots(figsize=self.settings['figsize'])
        sns.lineplot(x='epoch', y=self.name, data=data, ax=ax, hue='split_name', style='split_name',
                     markers=True, dashes=False, ci=None)

        max_num_epochs = 25
        num_epochs = len(self.epochs)
        if num_epochs < max_num_epochs:
            step_size = 1
        else:
            step_size = num_epochs // max_num_epochs + min(1, num_epochs % max_num_epochs)

        for i in range(0, data.shape[0]):
            epoch = data['epoch'].iloc[i]
            if not (epoch % step_size) or epoch == num_epochs - 1:
                ax.text(
                    epoch,
                    data[self.name].iloc[i] + 1e-5,
                    '{:.3f}'.format(np.round(data[self.name].iloc[i], 3)),
                    fontsize=self.settings['text_size']
                )

        if self.name.endswith('Accuracy'):
            #ax.set_ylim(0, 1.01)
            #ax.set_yticks(np.arange(0, 1.05, 0.1))
            ax.legend(fontsize=self.settings['legend_size'], loc='lower right')
        else:
            ax.set_ylim(0, None)
            ax.legend(fontsize=self.settings['legend_size'], loc='lower left')

        ax.set_xticks(sorted(list(self.epochs))[::step_size])
        ax.tick_params(labelsize=self.settings['label_size'])
        ax.grid(b=True, linestyle='--', alpha=self.settings['grid_alpha'])
        ax.set_xlabel('Epoch', fontsize=self.settings['axis_size'])
        ax.set_ylabel('')

        if self.name in ['Loss', 'Dice'] and self.domain is None:
            output_name = self.name.lower()
            title = self.name
        else:
            output_name = '-'.join(map(str.lower, (self.domain['name'], self.name)))
            title = '{}, {}'.format(self.name, self.domain['name'])

        ax.set_title(title, fontsize=self.settings['title_size'])

        output_path = os.path.join(output_dirs['plot'], output_name + '.' + self.settings['type'])
        plt.savefig(output_path, dpi=self.settings['dpi'])
        plt.close()

        data = {}
        variable = self.variable
        for split_name, epoch, value in zip(variable['split_name'], variable['epoch'], variable[self.name]):
            data.setdefault(split_name, {'epoch': [], 'value': []})
            data[split_name]['epoch'].append(epoch)
            data[split_name]['value'].append(str(value))

            if self.name not in ['Loss', 'Dice']:
                update_var_metrics(value, metrics, split_name, self.domain['name'], self.name)

        data = {'name': self.name, 'domain': self.domain, 'data': data}
        output_path = os.path.join(output_dirs['dump'], output_name + '.json')
        with open(output_path, 'w+') as output_file:
            json.dump(data, output_file, indent=4)
