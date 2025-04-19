import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Tekrarlanabilirlik
np.random.seed(42)

# 1) Veri Hazırlığı
# İstenen sıralama için:
#   ML modelleri: lr, xgb, mlp, svm, rf (bu sıraya dikkat ediniz)
ordered_algorithms = ["lr", "xgb", "mlp", "svm", "rf"]
sampling_methods = ["none", "smote", "rus"]  # İstenen sıralama: none, smote, rus
feature_reductions = ["none", "pca", "kbest"]  # İstenen sıralama: none, pca, kbest


# ML model kombinasyon etiketlerini oluşturacak fonksiyon:
def create_model_combo(a, s, f):
    parts = [a]  # Her zaman model ismi dahil
    if s.lower() != "none":
        parts.append(s)
    if f.lower() != "none":
        parts.append(f)
    return "-".join(parts).upper()


# 5 x 3 x 3 = 45 kombinasyon; örneğin "lr-none-none" yerine sadece "LR" oluşturulacak.
ml_combinations = [create_model_combo(a, s, f)
                   for a in ordered_algorithms
                   for s in sampling_methods
                   for f in feature_reductions]
num_combinations = len(ml_combinations)  # 45
# Aggregation ve metrikler
aggregations = ["Centralized", "FedAvg", "FedF1"]
metrics = ["Precision", "Recall", "Accuracy", "F1"]

# Dataset tanımı ve temel renkler (hex formatında)
datasets = ["TAIWAN", "HCDR", "HMEQ"]
ds_colors = {"TAIWAN": "#1f77b4", "HCDR": "#ff7f0e", "HMEQ": "#2ca02c"}

from notebooks.experiment_results import real_data

# 2) X Ekseni Konumlandırması – Metrik grupları bazında
# Her metrik grubu için:
#   "Precision": x = 0,1,2; "Recall": x = 4,5,6; "Accuracy": x = 8,9,10; "F1": x = 12,13,14
metric_base_map = {"Precision": 0, "Recall": 4, "Accuracy": 8, "F1": 12}

# Her (met, agg) kombinasyonu için x koordinatlarını hesaplayalım.
x_mapping = {}
for met in metrics:
    base = metric_base_map[met]
    for i, agg in enumerate(aggregations):
        x_mapping[(met, agg)] = base + i

# 3) X Ekseni için Major ve Minor Tick Etiketlerinin Oluşturulması
minor_positions = []
minor_labels = []
for met in metrics:
    base = metric_base_map[met]
    for i, agg in enumerate(aggregations):
        pos = base + i
        minor_positions.append(pos)
        minor_labels.append(agg)

major_positions = []
major_labels = []
for met in metrics:
    base = metric_base_map[met]
    major_positions.append(base + 1)
    major_labels.append(met)

# 4) Baloncuk (Bubble) Verilerinin Üretilmesi
bubble_scale = 1000
dataset_offsets = [-0.2, 0, 0.2]  # Aynı (met, agg) konumunda 3 dataset için

x_list = []
y_list = []
size_list = []
ds_list = []
metric_list = []
agg_list = []
ml_combo_list = []
metric_value_list = []

# Her ML model kombinasyonu, her metrik, her aggregation ve her dataset için gerçek değeri alıyoruz.
for y_idx, ml_combo in enumerate(ml_combinations):
    for met in metrics:
        for agg in aggregations:
            base_x = x_mapping[(met, agg)]
            for ds in datasets:
                # Gerçek veriden değer alalım. Eğer bir değer yoksa, varsayılan olarak 0.0 kullanılabilir.
                try:
                    val = real_data[ml_combo][met][agg][ds]
                except KeyError:
                    val = 0.0  # Gerçek verinizde eksik değer varsa bunu değiştirin.
                size_val = max(val * bubble_scale, 10)
                x_val = base_x + dataset_offsets[datasets.index(ds)]
                x_list.append(x_val)
                y_list.append(y_idx)
                size_list.append(size_val)
                ds_list.append(ds)
                metric_list.append(met)
                agg_list.append(agg)
                ml_combo_list.append(ml_combo)
                metric_value_list.append(val)


# 5) Renk Yoğunluğunun Ayarlanması
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_tuple):
    return '#{:02x}{:02x}{:02x}'.format(*rgb_tuple)


def interpolate_color(base_hex, t):
    """
    base_hex: temel renk (hex formatında)
    t: [0,1] arası; 0 => en hafif (beyaza yakın), 1 => temel renk (koyu)
    """
    r_base, g_base, b_base = hex_to_rgb(base_hex)
    r = int((1 - t) * 255 + t * r_base)
    g = int((1 - t) * 255 + t * g_base)
    b = int((1 - t) * 255 + t * b_base)
    return rgb_to_hex((r, g, b))


# Yeni sütun: CustomColor. t = MetricValue (0 => en hafif, 1 => temel renk (koyu))
custom_colors = []
for val, ds in zip(metric_value_list, ds_list):
    # t = val  # renk yoğunluğu değeri göre
    t = 1  # renk yoğunluğu yok
    custom_colors.append(interpolate_color(ds_colors[ds], t))

# DataFrame oluşturma
df = pd.DataFrame({
    "x": x_list,
    "y": y_list,
    "size": size_list,
    "Dataset": ds_list,
    "Metric": metric_list,
    "Aggregation": agg_list,
    "ML Model": ml_combo_list,
    "Result": metric_value_list,
    "CustomColor": custom_colors
})

# 6) Plotly Express ile Bubble Graph Çizimi
fig = px.scatter(
    df,
    x="x",
    y="y",
    size="size",
    hover_data=[
        "Metric",
        "Aggregation",
        "ML Model",
        "Result"
    ],
    color="Dataset"  # İlk etapta bu kullanılıyor; sonrasında renklendirme güncellenecek.
)

# Dataset bazındaki her trace'i güncelleyip marker renklerini, CustomColor sütunundaki değerlerle ayarlayalım.
for trace in fig.data:
    ds_name = trace.name
    mask = df["Dataset"] == ds_name
    trace.marker.color = df.loc[mask, "CustomColor"]

# X ekseni ayarlamaları: Minor tick'ler aggregation isimlerini (grid de var)
fig.update_xaxes(
    tickmode='array',
    tickvals=minor_positions,
    ticktext=minor_labels,
    showgrid=True,
    gridwidth=1,
    gridcolor='LightGray'
)

# Y ekseni: 45 ML model kombinasyonu. Tersi olacak şekilde (en alttaki model en üstte)
fig.update_yaxes(
    tickmode='array',
    tickvals=list(range(num_combinations)),
    ticktext=ml_combinations,
    showgrid=True,
    gridwidth=1,
    gridcolor='LightGray',
    autorange='reversed'
)

# Major tick etiketlerini annotation olarak ekleyelim (metrik grupları) – x ekseni altında.
annotations = []
for pos, label in zip(major_positions, major_labels):
    annotations.append(dict(
        x=pos,
        y=-0.05,
        xref="x",
        yref="paper",
        text=label,
        showarrow=False,
        font=dict(size=12)
    ))

fig.update_layout(
    annotations=annotations,
    title=dict(text="ML Model Results", x=0.5),
    xaxis_title="Aggregation Methodology (Minor ticks) – Metrics (Major annotations)",
    yaxis_title="ML Models",
    width=1600,
    height=1200,
    margin=dict(t=100, b=100),
    xaxis_title_standoff=60
)

# Baloncuk boyutlarının farkını daha belirgin kılmak için sizeref parametresi ayarlanıyor.
fig.update_traces(marker=dict(
    sizemode='area',
    sizeref=2. * max(df["size"]) / (20 ** 2)
))

fig.show()
