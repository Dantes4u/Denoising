import os


def get_dirs(input_dirs, dirname):
    output_dirs = {}
    for name, path in input_dirs.items():
        output_dirs[name] = os.path.join(path, dirname)
        os.makedirs(output_dirs[name], exist_ok=True)
    return output_dirs


def _update_metrics(metrics, split_name, domain_name, metric_name):
    metrics.setdefault(split_name, {})
    metrics[split_name].setdefault(domain_name, {})
    metrics[split_name][domain_name].setdefault(metric_name, {})


def _convert_metric_value(metric_value):
    if isinstance(metric_value, float):
        metric_value = '{:.3f}'.format(metric_value)
    else:
        metric_value = str(metric_value)
    return metric_value


def update_metrics(data, metrics, split_name, domain_name, metric_name):
    _update_metrics(metrics, split_name, domain_name, metric_name)
    for domain_value, metric_value in zip(data[domain_name], data[metric_name]):
        metrics[split_name][domain_name][metric_name][domain_value] = _convert_metric_value(metric_value)


def update_var_metrics(metric_value, metrics, split_name, domain_name, metric_name):
    _update_metrics(metrics, split_name, domain_name, metric_name)
    metrics[split_name][domain_name][metric_name]['all'] = _convert_metric_value(metric_value)

