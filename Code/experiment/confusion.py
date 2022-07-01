import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import confusion_matrix


def confusion_plot(data, split_name, domain, settings, output_dirs, **kwargs):
    domain_name = domain['name']
    pred_name = domain_name + '-pred'
    data = data.loc[:, [domain_name, pred_name]]
    data = data[data[domain_name] != settings['ignore_value']]
    if not data.empty:
        values = data[domain_name].unique().tolist() + data[pred_name].unique().tolist()
        values = [value for index, value in enumerate(domain['values']) if index in values]

        confusion = confusion_matrix(data[domain_name].values, data[pred_name].values)
        norm_confusion = 100 * confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)

        annotation = []
        for i in range(confusion.shape[0]):
            labels = []
            for value, norm in zip(confusion[i], norm_confusion[i]):
                if len(values) <= 10:
                    labels.append('{:d} ({:.2f}%)'.format(value, np.round(norm, 2)))
                else:
                    labels.append('{:.3f}'.format(np.round(norm / 100, 3)))
            annotation.append(labels)

        annotation = np.array(annotation)

        fig, ax = plt.subplots(figsize=settings['figsize'])
        ax = sns.heatmap(norm_confusion, ax=ax, annot=annotation, fmt='', xticklabels=values,
                         cmap=plt.cm.Blues, linecolor='black', linewidths=0.25, cbar=False)

        ax.set_title(', '.join((split_name, domain_name)), fontsize=settings['title_size'])
        ax.set_yticklabels(values, va='center', rotation=90, position=(0, 0.28))
        ax.set_xticklabels(values, va='center', rotation=0, position=(0, -0.01))
        ax.tick_params(labelsize=settings['label_size'])
        ax.grid(b=True, linestyle='--', alpha=settings['grid_alpha'])

        # y_max, y_min = plt.ylim()
        # y_max += 0.5
        # y_min -= 0.5
        # plt.ylim(y_max, y_min)

        output_name = '-'.join(map(str.lower, ('confusion', domain_name, split_name)))
        output_path = os.path.join(output_dirs['plot'], output_name)
        plt.savefig(output_path + '.' + settings['type'], dpi=settings['dpi'])
        plt.close()
