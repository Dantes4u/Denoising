import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from experiment import update_metrics


def accuracy_plot(data, split_name, domain, settings, output_dirs, metrics):
    domain_name = domain['name']
    pred_name = domain_name + '-pred'
    data = data.loc[:, [domain_name, pred_name]]
    data = data[data[domain_name] != settings['ignore_value']]
    if not data.empty:
        data.loc[:, 'Accuracy'] = (data[domain_name] == data[pred_name])
        data = data.astype({'Accuracy': 'int32'})
        data = data.drop(columns=pred_name)
        replace_map = dict(enumerate(domain['values']))
        data = data.replace({domain_name: replace_map})
        data = data.groupby([domain_name], as_index=False).mean()

        order = data[domain_name].unique()
        order = [value for value in domain['values'] if value in order]

        fig, ax = plt.subplots(figsize=settings['figsize'])
        palette = settings['colors'][domain_name] if 'colors' in settings else None
        sns.barplot(x=domain_name, y='Accuracy', data=data, ax=ax, order=order, saturation=settings['saturation'],
                    palette=palette)

        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=settings['label_size'])
        ax.grid(b=True, linestyle='--', alpha=settings['grid_alpha'])
        ax.set_xlabel('')
        ax.set_ylabel('Accuracy', fontsize=settings['axis_size'])
        ax.set_title('{}, {}'.format(split_name, domain_name), fontsize=settings['title_size'])

        for patch in ax.patches:
            height = max(0, patch.get_height())
            ax.text(
                patch.get_x() + patch.get_width() / 2.,
                height + 0.01,
                '{:.2f}'.format(np.round(height, 2)),
                ha='center',
                fontsize=settings['text_size']
            )

        output_name = '-'.join(map(str.lower, ('accuracy', split_name, domain_name)))

        output_path = os.path.join(output_dirs['plot'], output_name)
        plt.savefig(output_path + '.' + settings['type'], dpi=settings['dpi'])
        plt.close()

        update_metrics(data, metrics, split_name, domain_name, 'Accuracy')
        data = {'split_name': split_name, 'domain': domain, 'data': data.to_dict('list')}

        output_path = os.path.join(output_dirs['dump'], output_name + '.json')
        with open(output_path, 'w+') as output_file:
            json.dump(data, output_file, indent=4)
