from pathlib import Path

import pandas as pd
import plotly.express as px

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


# --- PLOTLY İLE GRAFİK ÇİZDİRME FONKSİYONU ---
def plot_full_comparison_plotly(metric_name):
    """
    Belirtilen metrik için tüm yaklaşımları, veri setlerini, modelleri ve
    örnekleme tekniklerini tek bir büyük grafikte karşılaştırır.
    (Sadece Feature Reduction == 'None' olan temel durumlar için)
    """

    # Sadece özellik azaltma uygulanmayanları filtrele
    df_filtered = df[df['Feature Reduction'] == 'None'].copy()

    if df_filtered.empty:
        print(f"--- UYARI: '{metric_name}' metriği için çizilecek veri bulunamadı. ---")
        return

    # Plotly Express ile 3x3'lük bir grid (matris) grafik oluşturalım
    fig = px.bar(
        df_filtered,
        x="ML Model",  # X ekseninde ML Modelleri
        y=metric_name,  # Y ekseni dinamik metrik değeri
        color="Data Sampling",  # Renk grupları Veri Örnekleme Teknikleri
        barmode="group",  # Barları yan yana grupla

        # --- ANAHTAR KISIM: GRAFİĞİ MATRİS HALİNE GETİRME ---
        facet_row="Aggregation",  # Satırlar Öğrenme Yaklaşımları olacak
        facet_col="Dataset",  # Sütunlar Veri Setleri olacak

        # Grafik estetiği ve etiketler
        category_orders={
            "Aggregation": ['Centralized', 'FedAvg', 'FedF1'],  # Satırların sırası
            "Dataset": ['DCCC', 'HCDR', 'HMEQ'],  # Sütunların sırası
            "ML Model": ['LR', 'MLP', 'SVM', 'XGB', 'RF'],  # X ekseni sırası
            "Data Sampling": ["None", "SMOTE", "RUS"]  # Renklerin sırası
        },
        color_discrete_map={
            "None": "skyblue",
            "SMOTE": "mediumseagreen",
            "RUS": "salmon"
        },
        labels={
            metric_name: f"{metric_name} Skoru",
            "ML Model": "Makine Öğrenmesi Modeli",
            "Data Sampling": "Veri Örnekleme Tekniği"
        },
        height=800,  # Grafiğin genel yüksekliği
        width=1400  # Grafiğin genel genişliği
    )

    # Genel başlığı ve düzenlemeleri yapalım
    fig.update_layout(
        title_text=f"Veri Örnekleme Tekniklerinin {metric_name} Skoruna Etkisi: Kapsamlı Karşılaştırma",
        title_x=0.5,
        font=dict(size=10),
        legend_title="Veri Örnekleme"
    )

    # Alt grafik başlıklarını daha sade hale getirelim
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Her bir alt grafiğin (facet) kendi y ekseni etiketini göstermesini sağlayalım
    # Bu, farklı satırlardaki değer aralıkları farklı olabileceğinden önemlidir.
    fig.update_yaxes(matches=None, showticklabels=True)

    # Grafiği gösterelim
    fig.show()


# --- ANA ÇALIŞTIRMA KISMI ---
# Hangi metrikler için grafik istediğinizi seçebilirsiniz
metrics_to_plot = ['Recall', 'F1', 'Precision', 'Accuracy']

# Her bir metrik için döngüye girip grafiği çizdirelim
for metric in metrics_to_plot:
    plot_full_comparison_plotly(metric_name=metric)
