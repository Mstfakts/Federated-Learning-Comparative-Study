def print_classification_report_from_dict(report_dict):
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            print(f"\nClass {label}:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value}")
        else:
            print(f"\n{label}: {metrics}")


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
