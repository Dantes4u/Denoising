import os
import json
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import use
from matplotlib.font_manager import FontProperties
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp


def roc_plot(data, split_name, domain, settings, output_dirs, **kwargs):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    domain_name = domain['name']
    data = data[data[domain_name] != -1]
    if not data.empty:
        num_values = len(domain['values'])
        if num_values != 2:
            bin_data = label_binarize(data[domain_name].values, classes=range(num_values))
        else:
            bin_data = np.array([[1, 0] if value == 0 else [0, 1] for value in data[domain_name].values])

        for i, value in enumerate(domain['values']):
            scores = data[domain_name + '-scores-{}'.format(i)].values
            fpr[value], tpr[value], _ = roc_curve(bin_data[:, i], scores)
            roc_auc[value] = auc(fpr[value], tpr[value])

        all_fpr = np.unique(np.concatenate([fpr[value] for value in domain['values']]))
        mean_tpr = np.zeros_like(all_fpr)
        mean_tpr_count = 0
        for value in domain['values']:
            if not np.isnan(tpr[value]).all():
                mean_tpr += interp(all_fpr, fpr[value], tpr[value])
                mean_tpr_count += 1
        mean_tpr /= mean_tpr_count

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    if not roc_plot:
        return

    colors = ['slateblue', 'seagreen', 'tomato', 'aqua', 'orchid', 'yellow', 'lime', 'darkslategray', 'goldenrod',
              'olive', 'dimgray', 'purple']
    use('AGG')
    font = FontProperties('Arial', size=24)
    plt.figure(figsize=(20, 20), dpi=101)
    ax = plt.subplot()

    ax.plot(100. * fpr["macro"], 100. * tpr["macro"],
            label='macro-average ROC curve (area = {0:0.3f})'.format(np.round(roc_auc["macro"], 3)), linestyle=':', lw=5,
            aa=True, alpha=0.9, color=colors[0])
    for idx, value in enumerate(domain['values']):
        if not np.isnan(tpr[value]).all():
            plt.plot(100. * fpr[value], 100. * tpr[value],
                     label='ROC curve of class {} (area = {:.3f})'.format(value, np.round(roc_auc[value], 3)), lw=4,
                     aa=True, alpha=0.9, color=colors[idx+1])

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([-1, 101])
    ax.set_ylim([-1, 101])
    ax.set_xlabel('False Positive Rate, %', fontproperties=font)
    ax.set_ylabel('True Positive Rate, %', fontproperties=font)
    ax.set_aspect('equal')
    ax.set_xticks(range(0, 101, 5), minor=False)
    ax.set_xticks(range(0, 101, 1), minor=True)
    ax.set_yticks(range(0, 101, 5), minor=False)
    ax.set_yticks(range(0, 101, 1), minor=True)
    ax.tick_params(axis='both', which='major', labelsize=24)
    ax.tick_params(axis='both', which='minor', labelsize=24)
    ax.set_title('ROC',  fontproperties=font)
    ax.legend(loc="lower right", prop=font)
    ax.grid(which='major', alpha=1.0, linewidth=2, linestyle='--')
    ax.grid(which='minor', alpha=0.5, linestyle='--')
    map(lambda l: l.set_fontproperties(font), ax.get_xticklabels() + ax.get_yticklabels())

    output_name = '-'.join(map(str.lower, ('roc', split_name, domain['name'])))

    output_path = os.path.join(output_dirs['plot'], output_name)
    plt.savefig(output_path + '.' + settings['type'], bbox_inches='tight')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)

    dump_data = {}
    for value in fpr:
        dump_data[value] = {'fpr': list(fpr[value]), 'tpr': list(tpr[value]), 'auc': roc_auc[value]}

    output_path = os.path.join(output_dirs['dump'], output_name)
    with open(output_path + '.json', 'w+') as output_file:
        json.dump(dump_data, output_file, indent=4)
