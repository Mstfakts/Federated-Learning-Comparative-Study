# Gerekli kütüphaneleri içe aktaralım
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
df["Feature Reduction"] = df["Feature Reduction"].replace("N", "None")
df["Data Sampling"] = df["Data Sampling"].replace("N", "None")


def plot_sampling_effect_all_models(metric_name, aggregation_name):
    """
    Belirtilen metrik ve öğrenme yaklaşımı için, tüm ML modellerinin
    veri örnekleme tekniklerinden nasıl etkilendiğini gösteren bir grafik çizer.
    (Sadece Feature Reduction == 'None' olan temel durumlar için)
    """

    # Sadece belirtilen öğrenme yaklaşımını ve özellik azaltma uygulanmayanları filtrele
    df_filtered = df[(df['Aggregation'] == aggregation_name) & (df['Feature Reduction'] == 'None')].copy()

    if df_filtered.empty:
        print(
            f"--- UYARI: '{aggregation_name}' yaklaşımında ve '{metric_name}' metriği için çizilecek veri bulunamadı. ---")
        return

    # Grafik için genel stil ve tema belirleyelim
    sns.set_theme(style="whitegrid")

    # Seaborn'un catplot fonksiyonunu kullanarak gruplandırılmış çubuk grafiği oluşturalım
    g = sns.catplot(
        data=df_filtered,
        kind="bar",
        x="ML Model",  # X ekseninde artık ML Modelleri var
        y=metric_name,  # Y ekseni dinamik metrik adı
        hue="Data Sampling",  # Gruplar Veri Örnekleme Teknikleri
        col="Dataset",  # Alt grafikler Veri Setlerine göre
        order=['LR', 'MLP', 'SVM', 'XGB', 'RF'],  # Modellerin sıralamasını sabitliyoruz
        palette={"None": "skyblue", "SMOTE": "mediumseagreen", "RUS": "salmon"},
        legend_out=True,
        height=6,
        aspect=1.2,  # Grafiğin genişlik/yükseklik oranını artırıyoruz
        errorbar=None
    )

    # Grafik başlıklarını ve etiketlerini dinamik olarak düzenleyelim
    g.fig.suptitle(f"'{aggregation_name}' Yaklaşımı İçin Veri Örneklemenin {metric_name} Skoruna Etkisi",
                   y=1.03, fontsize=16)
    g.set_axis_labels("Makine Öğrenmesi Modeli", f"{metric_name} Skoru")
    g.set_titles("Veri Seti: {col_name}")
    g.legend.set_title("Veri Örnekleme")

    # Barların üzerine değerlerini yazdıralım
    for axes in g.axes.flat:
        for p in axes.patches:
            axes.annotate(format(p.get_height(), '.2f'),
                          (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='center',
                          xytext=(0, 9),
                          textcoords='offset points',
                          fontsize=8)

    plt.ylim(0, 1.1)  # Değer etiketlerine yer açmak için limiti biraz artırdık
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# --- ANA ÇALIŞTIRMA KISMI ---
# Hangi öğrenme yaklaşımları için grafik istediğinizi seçebilirsiniz
aggregations_to_plot = ['Centralized', 'FedAvg', 'FedF1']

# Hangi metrikler için grafik istediğinizi seçebilirsiniz
metrics_to_plot = ['Recall', 'F1', 'Precision', 'Accuracy']

# Her bir öğrenme yaklaşımı ve metrik için grafikleri sırayla çizdirelim
for agg in aggregations_to_plot:
    for metric in metrics_to_plot:
        plot_sampling_effect_all_models(metric_name=metric, aggregation_name=agg)
