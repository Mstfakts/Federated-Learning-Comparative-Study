import os
import re
from collections import defaultdict


def print_classification_report_from_dict(report_dict, experiment_number=1):
    # Open the file in append mode to add new results
    with open(os.environ["result_filepath"], "a") as file:
        # Write experiment header
        experiment_header = f"\nEXPERIMENT #{experiment_number}:\n"
        print(experiment_header)
        file.write(experiment_header)

        # Write the classification report
        for label, metrics in report_dict.items():
            if isinstance(metrics, dict):
                class_report = f"\nClass {label}:"
                print(class_report)
                file.write(class_report + "\n")  # Write to file
                for metric_name, metric_value in metrics.items():
                    metric_report = f"  {metric_name}: {metric_value}"
                    print(metric_report)
                    file.write(metric_report + "\n")  # Write to file
            else:
                report = f"\n{label}: {metrics}"
                print(report)
                file.write(report + "\n")  # Write to file


def average_dict(d):
    average_metrics = {}
    for _, metrics in d:
        for key, value in metrics.items():
            if key in average_metrics:
                average_metrics[key].append(value)
            else:
                average_metrics[key] = [value]

    # Compute the average for each key
    average_metrics = {key: sum(values) / len(values) for key, values in average_metrics.items()}
    return average_metrics


def compute_averages(*metrics_list):
    average_metrics = defaultdict(dict)
    num_metrics = len(metrics_list)

    # Tüm metriklerin anahtarları üzerinden döngü
    for metrics in metrics_list:
        for key in metrics.keys():
            if key not in average_metrics:
                average_metrics[key] = defaultdict(float)

    # Her bir sınıf ve metrik için toplama işlemi
    for metrics in metrics_list:
        for key in metrics.keys():
            for metric in metrics[key]:
                average_metrics[key][metric] += metrics[key][metric]

    # Toplamların ortalamasını al
    for key in average_metrics.keys():
        for metric in average_metrics[key]:
            average_metrics[key][metric] /= num_metrics

    return average_metrics


def parse_experiment_data(file_content):
    # Düzenli ifade ile metrikleri ve değerleri yakalayacak desen
    metric_pattern = r"([A-Za-z0-9\s._-]+):\s*([\d.]+)"

    # Tüm metriklerin toplamlarını ve sayısını saklayacağımız sözlük
    metrics_sum = defaultdict(float)
    metrics_count = defaultdict(int)

    # Deneyleri yakalamak için
    experiments = file_content.split("EXPERIMENT #")

    for experiment in experiments[1:]:  # İlk öğe boş olabilir, atla
        for line in experiment.splitlines():
            match = re.match(metric_pattern, line.strip())
            if match:
                metric_name = match.group(1).strip()
                metric_value = float(match.group(2).strip())

                # Metrik değerini toplama ve sayısını artırma
                metrics_sum[metric_name] += metric_value
                metrics_count[metric_name] += 1

    # Ortalamaları hesapla
    average_metrics = {}
    for metric_name in metrics_sum:
        average_metrics[metric_name] = metrics_sum[metric_name] / metrics_count[metric_name]

    # Formatla
    aggregated_result = {
        'Class 0': {},
        'Class 1': {},
        'Class weighted avg': {},
        'Class macro avg': {},
        'Class test': {}
    }

    for key, value in average_metrics.items():
        if key.startswith('0.0'):
            new_key = key.split('_', 1)[1]
            aggregated_result['Class 0'][new_key] = value
        elif key.startswith('1.0'):
            new_key = key.split('_', 1)[1]
            aggregated_result['Class 1'][new_key] = value
        elif key.startswith('weighted avg'):
            new_key = key.split('_', 1)[1]
            aggregated_result['Class weighted avg'][new_key] = value
        elif key.startswith('macro avg'):
            new_key = key.split('_', 1)[1]
            aggregated_result['Class macro avg'][new_key] = value
        elif key.startswith('accuracy'):
            aggregated_result['Class test']['accuracy'] = value
        elif key.startswith('AUC'):
            continue
        else:
            aggregated_result[key] = value

    return aggregated_result


def parse_metrics(data_str):
    metric_pattern = r"(\w[\w-]*)\s*:\s*([\d.]+)"
    metrics_dict = defaultdict(dict)
    current_class = None
    for line in data_str.splitlines():
        # Sınıf başlığını yakala
        class_match = re.match(r"Class [\w\s]+:", line.strip())
        if class_match:
            current_class = line.strip()
            continue

        # Metriği yakala
        metric_match = re.match(metric_pattern, line.strip())
        if metric_match and current_class:
            metric_name = metric_match.group(1)
            metric_value = float(metric_match.group(2))
            metrics_dict[current_class][metric_name] = metric_value

    return metrics_dict


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='_'):
    result_dict = {}
    for key, value in d.items():
        keys = key.split(sep)
        current_dict = result_dict
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    return result_dict
