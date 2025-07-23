from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FILE_NAME = "TumDeneySonuclari.xlsx"
here = Path(__file__).resolve()
project_root = here.parents[2]
results_dir = project_root / "results" / "ml_pipeline_experiments" / "experimental_results" / FILE_NAME


def load_all_sheets(file_path):
    xls = pd.ExcelFile(file_path)
    df_list = []
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        df_list.append(df_sheet)

    df_res = pd.concat(df_list, ignore_index=True)
    return df_res


df = load_all_sheets(results_dir)
df["Dataset"] = df["Dataset"].replace("TAIWAN", "DCCC")
df_fl = df[df['Aggregation'].isin(['FedAvg', 'FedF1'])].copy()


# Her bir deney konfigürasyonunu (model + özellik azaltma + örnekleme)
def create_model_name(row):
    parts = [row['ML Model']]
    if row['Feature Reduction'] != 'None':
        parts.append(row['Feature Reduction'])
    if row['Data Sampling'] != 'None':
        parts.append(row['Data Sampling'])
    return '+'.join(parts)


df_fl['Model Configuration'] = df_fl.apply(create_model_name, axis=1)


def plot_metric_comparison(dataframe, metric_name):
    """
    Belirtilen metrik için FedAvg ve FedF1 karşılaştırma grafiğini çizer ve
    kantitatif analizi konsola yazdırır.

    Args:
        dataframe (pd.DataFrame): Karşılaştırma için hazırlanmış DataFrame.
        metric_name (str): Karşılaştırılacak metriğin adı ('F1', 'Precision', vb.).
    """

    pivot_df = dataframe.pivot_table(
        index=['Dataset', 'Model Configuration'],
        columns='Aggregation',
        values=metric_name
    ).reset_index()

    pivot_df.dropna(subset=['FedAvg', 'FedF1'], inplace=True)

    summary_lines = [f'--- FedF1 vs. FedAvg Quantitative Analysis ({metric_name}) ---']
    total_above, total_below, total_equal = 0, 0, 0

    for dataset_name in sorted(pivot_df['Dataset'].unique()):
        dataset_df = pivot_df[pivot_df['Dataset'] == dataset_name]
        above = (dataset_df['FedF1'] > dataset_df['FedAvg']).sum()
        below = (dataset_df['FedF1'] < dataset_df['FedAvg']).sum()
        equal = (dataset_df['FedF1'] == dataset_df['FedAvg']).sum()
        total = len(dataset_df)

        total_above += above
        total_below += below
        total_equal += equal

        summary_lines.append(f"\n- {dataset_name} ({total} experiments):")
        summary_lines.append(f"  FedF1 Superior   : {above}")
        summary_lines.append(f"  FedAvg Superior  : {below}")
        summary_lines.append(f"  Equal Performance: {equal}")

    summary_lines.append(f"\nOverall ({len(pivot_df)} experiments):")
    summary_lines.append(f"  FedF1 Superior   : {total_above}")
    summary_lines.append(f"  FedAvg Superior  : {total_below}")
    summary_lines.append(f"  Equal Performance: {total_equal}")
    summary_lines.append("--------------------------------------------------")

    summary_text = '\n'.join(summary_lines)
    print(summary_text)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 5))
    scatter_plot = sns.scatterplot(
        data=pivot_df,
        x='FedAvg',
        y='FedF1',
        hue='Dataset',
        palette={'DCCC': 'purple', 'HCDR': 'red', 'HMEQ': 'green'},
        s=80,
        alpha=0.7,
        edgecolor='w',
        linewidth=0.5
    )

    min_val = min(pivot_df['FedAvg'].min(), pivot_df['FedF1'].min()) * 0.95
    max_val = max(pivot_df['FedAvg'].max(), pivot_df['FedF1'].max()) * 1.05

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Equal Performance (y=x)')

    #plt.title(f'{metric_name} Comparison of FedF1 and FedAvg', fontsize=20, pad=10)
    plt.xlabel(f'FedAvg {metric_name}', fontsize=16)
    plt.ylabel(f'FedF1 {metric_name}', fontsize=16)

    plt.legend(title='Dataset', title_fontsize='14', fontsize='11', loc='lower right')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"fedf1_vs_fedavg_{metric_name}.png", dpi=300)
    plt.show()


# --- ANA ÇALIŞTIRMA KISMI ---
# Analiz edilecek metriklerin listesini tanımlayalım
metrics_to_plot = ['F1', 'Precision', 'Recall', 'Accuracy']

# Her bir metrik için döngüye girip grafiği çizdirelim
for metric in metrics_to_plot:
    plot_metric_comparison(df_fl, metric)
