import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.pyplot import figure, legend, plot, subplot
from matplotlib import use

from sklearn.metrics import precision_recall_curve


def pr_plot(data, split_name, domain, settings, output_dirs, **kwargs):
    num_values = domain['length']
    if num_values != 2:
        return

    domain_name = domain['name']
    data = data[data[domain_name] != settings['ignore_value']]
    if data.empty:
        return

    precision, recall, thresholds = precision_recall_curve(
        data[domain_name].values,
        data[domain_name + '-scores-1'].values
    )
    for i in range(len(precision)):
        precision[i] = np.max(precision[:i+1])
    average_precision = -np.sum(np.diff(recall) * np.array(precision)[:-1])

    use('AGG')
    color = 'slateblue'
    y_values = [100. * x for x in precision]
    x_values = [100. * x for x in recall]

    font = FontProperties('Arial', size=24)
    figure(figsize=(20, 20), dpi=101)
    axes = subplot()
    axes.set_xlim([-1, 101])
    axes.set_ylim([-1, 101])
    axes.set_xlabel('Recall, %', fontproperties=font)
    axes.set_ylabel('Precision, %', fontproperties=font)
    axes.set_aspect('equal')
    axes.set_xticks(range(0, 101, 5), minor=False)
    axes.set_xticks(range(0, 101, 1), minor=True)
    axes.set_yticks(range(0, 101, 5), minor=False)
    axes.set_yticks(range(0, 101, 1), minor=True)
    axes.tick_params(axis='both', which='major', labelsize=24)
    axes.tick_params(axis='both', which='minor', labelsize=24)
    axes.grid(which='major', alpha=1.0, linewidth=2, linestyle='--')
    axes.grid(which='minor', alpha=0.5, linestyle='--')
    axes.set_title('Precision-Recall for {}, {}'.format(split_name, domain_name), fontproperties=font)

    plot(x_values, y_values, '-', color=color, aa=True, alpha=0.9, linewidth=5,
         label=f'AP: {np.round(average_precision, 4):.04f}')
    map(lambda l: l.set_fontproperties(font), axes.get_xticklabels() + axes.get_yticklabels())
    legend(loc='lower center', prop=font)

    output_name = '-'.join(map(str.lower, ('pr', split_name, domain['name'])))
    output_path = os.path.join(output_dirs['plot'], output_name)
    plt.savefig(output_path + '.' + settings['type'], bbox_inches='tight')
    plt.close()

    precision, recall, thresholds = (list(map(float, values)) for values in map(list, (precision, recall, thresholds)))
    data = {
        'split name': split_name, 'domain': domain,
        'precision': precision, 'recall': recall, 'thresholds': thresholds
    }
    output_path = os.path.join(output_dirs['dump'], output_name + '.json')
    with open(output_path, 'w+') as output_file:
        json.dump(data, output_file, indent=4)

