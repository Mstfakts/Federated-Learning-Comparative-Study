import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

labels = ["Central ML", "FL with FedAvg", "FL with FedF1"]

central_ml_accuracy = [0.81,0.82,0.76,0.68,0.82,0.80,0.81,0.81,0.70,0.81,0.81,0.82,0.82,0.77,0.82,0.69,0.79,0.55,0.57,0.77,0.68,0.76,0.72,0.69,0.78,0.77,0.80,0.77,0.77,0.80,0.67,0.75,0.71,0.69,0.76,0.66,0.75,0.73,0.70,0.77,0.77,0.76,0.77,0.77,0.78]
fedavg_accuracy = [0.80,0.82,0.81,0.68,0.81,0.79,0.81,0.80,0.69,0.82,0.79,0.81,0.82,0.77,0.82,0.71,0.76,0.68,0.62,0.79,0.62,0.66,0.64,0.64,0.77,0.76,0.71,0.75,0.78,0.80,0.53,0.76,0.73,0.69,0.75,0.62,0.65,0.60,0.62,0.75,0.50,0.76,0.76,0.77,0.78]
fedf1_accuracy = [0.81,0.81,0.68,0.68,0.81,0.80,0.81,0.80,0.69,0.81,0.81,0.81,0.82,0.78,0.82,0.70,0.71,0.55,0.62,0.79,0.63,0.68,0.61,0.64,0.78,0.74,0.67,0.75,0.77,0.80,0.53,0.76,0.73,0.69,0.75,0.61,0.68,0.58,0.64,0.76,0.48,0.76,0.76,0.78,0.77]

central_ml_f1 = [0.35,0.46,0.23,0.46,0.47,0.31,0.43,0.47,0.47,0.44,0.32,0.48,0.46,0.51,0.46,0.45,0.53,0.42,0.41,0.52,0.47,0.52,0.50,0.47,0.53,0.51,0.53,0.52,0.52,0.53,0.47,0.53,0.50,0.48,0.53,0.46,0.52,0.51,0.47,0.53,0.51,0.53,0.53,0.51,0.53]
fedavg_f1 = [0.26,0.46,0.46,0.46,0.42,0.22,0.39,0.43,0.45,0.42,0.18,0.39,0.49,0.50,0.46,0.47,0.50,0.47,0.44,0.52,0.39,0.40,0.33,0.39,0.50,0.50,0.49,0.50,0.50,0.51,0.40,0.52,0.50,0.47,0.52,0.38,0.41,0.37,0.39,0.50,0.39,0.51,0.52,0.50,0.51]
fedf1_f1 = [0.34,0.40,0.34,0.46,0.42,0.29,0.38,0.44,0.46,0.40,0.37,0.40,0.49,0.51,0.46,0.47,0.50,0.40,0.43,0.52,0.40,0.42,0.33,0.41,0.51,0.50,0.47,0.51,0.51,0.52,0.41,0.52,0.49,0.47,0.52,0.38,0.44,0.36,0.39,0.52,0.39,0.52,0.52,0.50,0.51]

central_ml_precision = [0.71, 0.66, 0.44, 0.38, 0.67, 0.67, 0.61, 0.63, 0.39, 0.63, 0.73, 0.69, 0.68, 0.49, 0.67, 0.37, 0.53, 0.29, 0.29, 0.49, 0.37, 0.46, 0.42, 0.38, 0.50, 0.48, 0.54, 0.49, 0.49, 0.54, 0.37, 0.46, 0.40, 0.39, 0.47, 0.35, 0.45, 0.43, 0.39, 0.47, 0.48, 0.50, 0.48, 0.49, 0.50]
fedavg_precision = [0.71, 0.68, 0.63, 0.37, 0.65, 0.70, 0.64, 0.58, 0.37, 0.66, 0.75, 0.68, 0.64, 0.49, 0.64, 0.39, 0.46, 0.37, 0.33, 0.53, 0.30, 0.34, 0.28, 0.31, 0.48, 0.47, 0.41, 0.45, 0.50, 0.55, 0.28, 0.49, 0.43, 0.38, 0.45, 0.30, 0.33, 0.29, 0.30, 0.44, 0.30, 0.47, 0.46, 0.47, 0.46]
fedf1_precision = [0.69, 0.66, 0.32, 0.37, 0.67, 0.67, 0.63, 0.59, 0.37, 0.63, 0.71, 0.69, 0.63, 0.50, 0.65, 0.39, 0.42, 0.29, 0.32, 0.52, 0.32, 0.35, 0.27, 0.33, 0.49, 0.46, 0.39, 0.45, 0.49, 0.54, 0.29, 0.48, 0.42, 0.38, 0.45, 0.29, 0.37, 0.28, 0.31, 0.46, 0.29, 0.48, 0.47, 0.50, 0.47]

central_ml_recall = [0.23,0.36,0.19,0.62,0.37,0.20,0.33,0.38,0.61,0.34,0.21,0.36,0.34,0.54,0.36,0.59,0.53,0.74,0.70,0.56,0.64,0.59,0.62,0.62,0.55,0.54,0.53,0.57,0.55,0.52,0.67,0.64,0.65,0.64,0.62,0.67,0.62,0.64,0.60,0.59,0.54,0.57,0.59,0.53,0.57]
fedavg_recall = [0.16,0.35,0.37,0.64,0.32,0.13,0.28,0.36,0.61,0.31,0.11,0.29,0.40,0.53,0.37,0.60,0.56,0.68,0.68,0.51,0.56,0.51,0.41,0.54,0.53,0.56,0.62,0.59,0.52,0.49,0.74,0.57,0.63,0.65,0.64,0.56,0.53,0.54,0.55,0.60,0.74,0.58,0.61,0.55,0.58]
fedf1_recall = [0.23,0.30,0.39,0.63,0.32,0.18,0.28,0.36,0.61,0.30,0.25,0.30,0.41,0.53,0.36,0.60,0.65,0.71,0.68,0.53,0.56,0.52,0.45,0.55,0.54,0.58,0.66,0.59,0.55,0.50,0.75,0.58,0.61,0.65,0.63,0.54,0.55,0.55,0.55,0.60,0.75,0.59,0.61,0.53,0.57]

# DataFrame oluşturma
metrics = ['Accuracy', 'F1', 'Precision', 'Recall']
values = central_ml_accuracy + fedavg_accuracy + fedf1_accuracy + central_ml_f1 + fedavg_f1 + fedf1_f1 + central_ml_precision + fedavg_precision + fedf1_precision + central_ml_recall + fedavg_recall + fedf1_recall
methods = [labels[0]] * len(central_ml_accuracy) + [labels[1]] * len(fedavg_accuracy) + [labels[2]] * len(
    fedf1_accuracy)
methods *= 4
metric_labels = [metrics[0]] * len(central_ml_accuracy + fedavg_accuracy + fedf1_accuracy) + [metrics[1]] * len(
    central_ml_f1 + fedavg_f1 + fedf1_f1) + [metrics[2]] * len(
    central_ml_precision + fedavg_precision + fedf1_precision) + [metrics[3]] * len(
    central_ml_recall + fedavg_recall + fedf1_recall)

df = pd.DataFrame({'Metric': metric_labels, 'Value': values, 'Method': methods})

# Subfigürler oluşturma
fig = make_subplots(rows=2, cols=2, subplot_titles=('(a) Precision', '(b) Recall', '(c) F1', '(d) Accuracy'),
                    horizontal_spacing=0.04,
                    #vertical_spacing=0.16
                    )

box_width = 0.2

# Precision için violin grafiği
fig_precision = px.violin(df[df['Metric'] == 'Precision'], x="Method", y="Value", color="Method", box=True, points="all")
for trace in fig_precision.data:
    trace.update(showlegend=True, box_visible=True, box_line_width=1, box_width=box_width)
    fig.add_trace(trace, row=1, col=1)

# Recall için violin grafiği
fig_recall = px.violin(df[df['Metric'] == 'Recall'], x="Method", y="Value", color="Method", box=True, points="all")
for trace in fig_recall.data:
    trace.update(showlegend=False, box_visible=True, box_line_width=1, box_width=box_width)
    fig.add_trace(trace, row=1, col=2)

# F1 için violin grafiği
fig_f1 = px.violin(df[df['Metric'] == 'F1'], x="Method", y="Value", color="Method", box=True, points="all")
for trace in fig_f1.data:
    trace.update(showlegend=False, box_visible=True, box_line_width=1, box_width=box_width)
    fig.add_trace(trace, row=2, col=1)

# Accuracy için violin grafiği
fig_accuracy = px.violin(df[df['Metric'] == 'Accuracy'], x="Method", y="Value", color="Method", box=True, points="all")
for trace in fig_accuracy.data:
    trace.update(showlegend=False, box_visible=True, box_line_width=1, box_width=box_width)
    fig.add_trace(trace, row=2, col=2)

# Yazı boyutlarını büyütme ve ortak legend ekleme
fig.update_layout(
    # title_font=dict(size=20),
    xaxis_title_font=dict(size=24),
    yaxis_title_font=dict(size=24),
    legend_font=dict(size=24),
    # xaxis_tickfont=dict(size=24),
    # yaxis_tickfont=dict(size=24),
    height=800,
    width=1400,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.05,
        xanchor="center",
        x=0.5
    )
)
# X eksenindeki etiketleri kaldırma
for i in range(1, 5):
    fig['layout'][f'xaxis{i}']['title'] = None
    fig['layout'][f'xaxis{i}']['showticklabels'] = False

# Alt başlıkları kalın yapma
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=24, color='black')

fig.show()
