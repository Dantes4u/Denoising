import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from experiment import update_metrics


def dist_plot(data, split_name, domain, settings, output_dirs, metrics):
    domain_name = domain['name']
    #data = data[data[domain_name] != settings['ignore_value']]
    data_size = len(data)
    print(data)
    if not data.empty:
        fig, ax = plt.subplots(figsize=settings['figsize'])
        bin_field = '{}-bin'.format(domain_name)
        if bin_field in data:
            field_name = bin_field
            data = data.loc[:, [field_name]]
            data = data.groupby([field_name]).size().to_frame('Distribution').reset_index()

            values = data[field_name].unique()
            values = sorted(values, key=lambda x: list(map(int, x.split(':'))))

            sns.barplot(x=field_name, y='Distribution', data=data, order=values, color=settings['colors'][domain_name],
                        ax=ax, ci=None)
        else:
            field_name = domain_name
            data = data.loc[:, [field_name]]
            replace_map = dict(enumerate(domain['values']))
            data = data.replace({field_name: replace_map})

            values = data[field_name].unique()
            values = [value for value in domain['values'] if value in values]

            palette = settings['colors'][domain_name] if 'colors' in settings else None
            sns.countplot(x=field_name, data=data, ax=ax, order=values, saturation=settings['saturation'],
                          palette=palette)
            ax.grid(b=True, linestyle='--', alpha=settings['grid_alpha'])

        ax.set_title(split_name + ', ' + domain_name, fontsize=settings['title_size'])
        ax.set_xlabel('{}, total {:d}'.format(domain_name, data_size), fontsize=settings['axis_size'])
        ax.set_ylabel('Number', fontsize=settings['axis_size'])
        ax.tick_params(labelsize=settings['label_size'])
        ax.grid(b=True, linestyle='--', alpha=settings['grid_alpha'])
        ax.set_ylim(0, None)

        for patch in ax.patches:
            height = int(max(0, patch.get_height()))
            ax.text(
                patch.get_x() + patch.get_width() / 2.,
                height + 0.01,
                '{:d}'.format(height),
                ha='center',
                fontsize=settings['text_size']
            )

        output_name = '-'.join(map(str.lower, ('dist', domain_name, split_name)))

        output_path = os.path.join(output_dirs['plot'], output_name)
        plt.savefig(output_path + '.' + settings['type'], dpi=settings['dpi'])
        plt.close()

        if bin_field not in data:
            data = data[domain_name].value_counts()
            data = data.rename_axis(domain_name).reset_index(name='Distribution')

        update_metrics(data, metrics, split_name, field_name, 'Distribution')
        data = {'split_name': split_name, 'domain': domain, 'data': data.to_dict('list')}

        output_path = os.path.join(output_dirs['dump'], output_name + '.json')
        with open(output_path, 'w+') as output_file:
            json.dump(data, output_file, indent=4)
