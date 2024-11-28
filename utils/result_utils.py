import os
import re
from collections import defaultdict

os.environ["config_file"] = "xgboost"
from configs.config import config

print(f"###### SELECTED RESULTS: {os.environ['config_file']} ######")

metric_pattern = r"(\w[\w-]*)\s*:\s*([\d.]+)"


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

    return average_metrics


# Sınıf ve metrikleri ayrıştırmak için fonksiyon
def parse_metrics(data_str):
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


# Dosya içeriğini oku ve deneyleri ayır
with open("C:\\Users\B3LAB\PycharmProjects\FL-Benchmark\experiments\hyperparameter_tuning\\xgboost\\results.txt", 'r') as file:
    content = file.read()

if os.environ["config_file"] == "xgboost":
    average_metrics = parse_experiment_data(content)

    print("Metriklerin Ortalamaları:")
    for metric_name, avg_value in average_metrics.items():
        print(f"{metric_name}: {avg_value}")

else:
    experiments = content.split("EXPERIMENT #")[1:]  # İlk boşluğu silmek için [1:] kullanılır

    # Her bir deneyi değişkenlere ata
    veri1 = parse_metrics(experiments[0])
    veri2 = parse_metrics(experiments[1])
    veri3 = parse_metrics(experiments[2])
    veri4 = parse_metrics(experiments[3])
    veri5 = parse_metrics(experiments[4])
    veri6 = parse_metrics(experiments[5])
    veri7 = parse_metrics(experiments[6])
    # veri8 = parse_metrics(experiments[7])
    # veri9 = parse_metrics(experiments[8])
    # veri10 = parse_metrics(experiments[9])

    # Ortalamaları hesapla
    averaged_metrics = compute_averages(veri1,
                                        veri2,
                                        veri3,
                                        veri4,
                                        veri5,
                                        veri6,
                                        veri7,
                                        # veri8,
                                        # veri9,
                                        # veri10
                                        )

    # Sonuçları yazdırma
    print("Metriklerin Ortalamaları:")
    for cls, metrics in averaged_metrics.items():
        print(f"{cls}:")
        for metric, avg in metrics.items():
            print(f"  {metric}: {avg}")
