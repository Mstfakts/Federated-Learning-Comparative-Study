def print_classification_report_from_dict(report_dict):
    for label, metrics in report_dict.items():
        if isinstance(metrics, dict):
            print(f"\nClass {label}:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value}")
        else:
            print(f"\n{label}: {metrics}")
