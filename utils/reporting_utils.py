def print_classification_report_from_dict(report_dict, experiment_number=1):
    # Open the file in append mode to add new results
    with open("results.txt", "a") as file:
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
