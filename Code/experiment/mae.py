import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from experiment import update_metrics


def mae_plot(data, split_name, domain, settings, output_dirs, metrics):
    domain_name = domain['name']
    pred_name = domain_name + '-pred'
    bin_field = '{}-bin'.format(domain_name)
    if bin_field in data:
        data = data.loc[:, [domain_name, pred_name, bin_field]]
    else:
        data = data.loc[:, [domain_name, pred_name]]
    data = data[data[domain_name] != settings['ignore_value']]
    if not data.empty:
        data['MAE'] = np.abs(data[domain_name] - data[pred_name])
        data = data.astype({'MAE': 'int32'})
        if bin_field in data:
            field_name = bin_field
            values = data[field_name].unique()
            values = sorted(values, key=lambda x: list(map(int, x.split(':'))))
        else:
            field_name = domain_name
            values = data[field_name].unique()
            values = sorted(values)

        data = data.loc[:, [field_name, 'MAE']]
        data = data.groupby([field_name]).mean().reset_index()

        fig, ax = plt.subplots(figsize=settings['figsize'])
        sns.barplot(x=field_name, y='MAE', data=data, order=values, color=settings['colors'][domain_name],
                    ax=ax, ci=None)

        ax.set_title('{}, {}'.format(split_name, domain_name), fontsize=settings['title_size'])
        ax.set_xlabel('')
        ax.set_ylabel('MAE', fontsize=settings['axis_size'])
        ax.tick_params(labelsize=settings['label_size'])
        ax.grid(b=True, linestyle='--', alpha=settings['grid_alpha'])
        ax.set_ylim(0, None)

        for patch in ax.patches:
            height = max(0, patch.get_height())
            ax.text(
                patch.get_x() + patch.get_width() / 2.,
                height + 0.01,
                '{:.2f}'.format(np.round(height, 2)),
                ha='center',
                fontsize=settings['text_size']
            )

        output_name = '-'.join(map(str.lower, ('mae', domain_name, split_name)))
        output_path = os.path.join(output_dirs['plot'], output_name)
        plt.savefig(output_path + '.' + settings['type'], dpi=settings['dpi'])
        plt.close()

        update_metrics(data, metrics, split_name, field_name, 'MAE')
        data = {'split_name': split_name, 'domain': domain, 'data': data.to_dict('list')}

        output_path = os.path.join(output_dirs['dump'], output_name + '.json')
        with open(output_path, 'w+') as output_file:
            json.dump(data, output_file, indent=4)
