# Gerekli kütüphaneleri içe aktaralım
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
df["Feature Reduction"] = df["Feature Reduction"].replace("N", "None")
df["Data Sampling"] = df["Data Sampling"].replace("N", "None")


# --- PERFORMANS DEĞİŞİM ISI HARİTASI FONKSİYONU ---
def create_delta_heatmap(metric_name='F1'):
    """
    Ön işleme tekniklerinin temel modele göre F1-Skorundaki yüzdesel
    değişimini gösteren 3x3'lük bir ısı haritası gridi oluşturur.
    """

    # Yeni birleşik ön işleme sütunu oluşturalım
    def get_preprocessing_name(row):
        fr = row['Feature Reduction']
        ds = row['Data Sampling']
        if fr == 'None' and ds == 'None': return 'Base (None)'
        if fr == 'None': return ds
        if ds == 'None': return fr
        return f"{ds}+{fr}"

    df['Preprocessing'] = df.apply(get_preprocessing_name, axis=1)

    # Temel skorları (hiçbir ön işleme uygulanmamış) alalım
    base_scores = df[df['Preprocessing'] == 'Base (None)'].copy()
    base_scores = base_scores.set_index(['Dataset', 'Aggregation', 'ML Model'])[[metric_name]]
    base_scores.rename(columns={metric_name: 'Base_Score'}, inplace=True)

    # Tüm skorları temel skorlarla birleştirelim
    merged_df = df.join(base_scores, on=['Dataset', 'Aggregation', 'ML Model'])

    # Yüzdesel değişimi hesaplayalım
    merged_df['Delta (%)'] = ((merged_df[metric_name] - merged_df['Base_Score']) / merged_df['Base_Score']) * 100
    merged_df.replace([np.inf, -np.inf], 0, inplace=True)

    # --- ORTAK RENK SKALASI İÇİN GLOBAL MİN/MAX HESAPLAMA ---
    vmin, vmax = -100, 100

    # Grafik için sıralamaları tanımlayalım
    aggregations = ['Centralized', 'FedAvg', 'FedF1']
    datasets = ['DCCC', 'HCDR', 'HMEQ']
    ml_models = ['LR', 'MLP', 'SVM', 'XGB', 'RF']

    preprocessing_order = [
        "RUS",
        "SMOTE",
        "PCA",
        "kBest",
        "RUS+PCA",
        "RUS+kBest",
        "SMOTE+PCA",
        "SMOTE+kBest"
    ]

    # 3x3'lük bir figür oluşturalım
    fig, axes = plt.subplots(
        nrows=len(aggregations),
        ncols=len(datasets),
        figsize=(14, 10),
        sharey=True
    )


    fig.suptitle(f"Impact of Preprocessing Techniques on {metric_name} Score, Percentage Change from Baseline", fontsize=24, y=0.99)

    for i, agg in enumerate(aggregations):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]

            plot_data = merged_df[
                (merged_df['Aggregation'] == agg) &
                (merged_df['Dataset'] == ds) &
                (merged_df['Preprocessing'] != 'Base (None)')
                ].pivot_table(
                index='ML Model',
                columns='Preprocessing',
                values='Delta (%)',
                fill_value=0
            )

            plot_data = plot_data.reindex(index=ml_models, columns=preprocessing_order)

            sns.heatmap(
                plot_data,
                annot=True, fmt=".1f", linewidths=.5,
                cmap="RdYlGn", center=0, ax=ax,
                cbar=False,
                vmin=vmin, vmax=vmax,
                annot_kws={
                    "rotation": 90,  # metni 45 derece döndür
                    "ha": "center",  # yatay hizalama
                    "va": "center",  # dikey hizalama
                    "fontweight": "bold"
                }
            )

            if i == 0:
                ax.set_title(ds, fontsize=16, pad=15)
            else:
                ax.set_title("")

            if j == 0:
                ax.set_ylabel(agg, fontsize=16, labelpad=20)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")

            if i == len(aggregations) - 1:
                ax.tick_params(axis='x', labelbottom=True, labelrotation=90, labelsize=12)
                for lbl in ax.get_xticklabels():
                    lbl.set_ha('right')
            else:
                ax.tick_params(axis='x', labelbottom=False)

            ax.tick_params(
                axis='y',
                labelsize=12
            )



    # fig.text(0.5, 0.04, 'Ön İşleme Kombinasyonu', ha='center', va='center', fontsize=18)

    fig.subplots_adjust(left=0.15, right=0.92, top=0.9, bottom=0.3, hspace=0.3, wspace=0.1)
    plt.tight_layout()
    plt.savefig(f"preprocessing_heatmap_{metric_name}_90.png", dpi=300)
    plt.show()


# --- ANA ÇALIŞTIRMA KISMI ---
metrics_to_plot = ['Recall', 'F1', 'Precision', 'Accuracy']
for metric in metrics_to_plot:
    create_delta_heatmap(metric_name=metric)
