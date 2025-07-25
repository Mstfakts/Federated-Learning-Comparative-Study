# Gerekli kütüphaneleri içe aktaralım
# Plotly'nin kurulu olması gerekir: pip install plotly pandas openpyxl
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


# --- BAĞLANTILI NOKTA GRAFİĞİ ÇİZDİRME FONKSİYONU ---
def plot_sampling_effect_dot_plot(metric_name):
    """
    Tüm senaryoları içeren bir grid üzerinde, veri örneklemenin etkisini
    bağlantılı nokta grafiği ile gösterir.
    """
    df_filtered = df[df['Feature Reduction'] == 'None'].copy()

    aggregations = sorted(df_filtered['Aggregation'].unique())
    datasets = sorted(df_filtered['Dataset'].unique())

    # 3x3'lük bir alt grafik yapısı oluşturalım
    fig = make_subplots(
        rows=len(aggregations),
        cols=len(datasets),
        subplot_titles=[f"{ds} - {agg}" for agg in aggregations for ds in datasets],
        shared_yaxes=True  # Y eksenini modeller için ortak tutalım
    )

    ml_models = ['LR', 'MLP', 'SVM', 'XGB', 'RF']
    colors = {"None": "skyblue", "SMOTE": "mediumseagreen", "RUS": "salmon"}

    for i, agg in enumerate(aggregations):
        for j, ds in enumerate(datasets):
            # Her bir alt grafiğe ait veriyi filtrele
            plot_data = df_filtered[(df_filtered['Aggregation'] == agg) & (df_filtered['Dataset'] == ds)]

            if plot_data.empty:
                continue

            # Her bir ML modeli için yatay çizgileri (dambılları) çiz
            for model_name in ml_models:
                model_data = plot_data[plot_data['ML Model'] == model_name]
                none_val = model_data[model_data['Data Sampling'] == 'None'][metric_name].values
                smote_val = model_data[model_data['Data Sampling'] == 'SMOTE'][metric_name].values
                rus_val = model_data[model_data['Data Sampling'] == 'RUS'][metric_name].values

                # Değerler varsa çizgi ve noktaları ekle
                if len(none_val) > 0 and (len(smote_val) > 0 or len(rus_val) > 0):
                    x_vals = [
                        none_val[0] if len(none_val) > 0 else None,
                        smote_val[0] if len(smote_val) > 0 else None,
                        rus_val[0] if len(rus_val) > 0 else None
                    ]
                    # Yatay çizgiyi ekle
                    fig.add_trace(go.Scatter(
                        x=[min(x for x in x_vals if x is not None), max(x for x in x_vals if x is not None)],
                        y=[model_name, model_name],
                        mode='lines',
                        line=dict(color='lightgrey', width=1),
                        showlegend=False
                    ), row=i + 1, col=j + 1)

            # Her bir örnekleme tekniği için noktaları (scatter) çiz
            for sampling_type in ["None", "SMOTE", "RUS"]:
                sampling_data = plot_data[plot_data['Data Sampling'] == sampling_type]
                fig.add_trace(go.Scatter(
                    x=sampling_data[metric_name],
                    y=sampling_data['ML Model'],
                    mode='markers',
                    marker=dict(color=colors[sampling_type], size=10, line=dict(width=1, color='DarkSlateGrey')),
                    name=sampling_type,
                    # Legend'i sadece ilk grafikte göstererek tekrarı önle
                    showlegend=(i == 0 and j == 0)
                ), row=i + 1, col=j + 1)

    # Genel grafik düzenlemeleri
    fig.update_layout(
        title_text=f"Veri Örnekleme Tekniklerinin {metric_name} Skoruna Etkisi (Bağlantılı Nokta Grafiği)",
        height=800,
        width=1200,
        title_x=0.5,
        legend_title="Veri Örnekleme"
    )
    fig.update_xaxes(range=[0, 1.05])

    # Alt başlıklar için metinleri temizle
    for annotation in fig['layout']['annotations']:
        annotation['text'] = annotation['text'].replace('Dataset=', '').replace('Aggregation=', '')

    fig.show()


# --- ANA ÇALIŞTIRMA KISMI ---
metrics_to_plot = ['Recall', 'F1', 'Precision', 'Accuracy']
for metric in metrics_to_plot:
    plot_sampling_effect_dot_plot(metric)